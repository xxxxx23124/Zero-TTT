"""Accumulated BF16 training with fast/slow weights and fault-stop semantics."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from zero_ttt.config import ExperimentConfig
from zero_ttt.model.transformer import PolicyValueTransformer
from zero_ttt.replay.sampler import ReplaySampler, SampledBatch
from zero_ttt.training.checkpoint import CheckpointManager, checkpoint_metadata
from zero_ttt.training.ema import update_slow_weights
from zero_ttt.training.losses import TrainingTargets, compute_losses


@dataclass(slots=True)
class TrainerState:
    optimizer_step: int = 0
    samples_seen: int = 0
    ema_pending_samples: int = 0
    last_published_step: int = 0


@dataclass(frozen=True, slots=True)
class StepMetrics:
    step: int
    learning_rate: float
    hypernet_scale: float
    total_loss: float
    policy_loss: float
    value_loss: float
    ownership_loss: float
    score_loss: float
    gradient_norm: float


class Trainer:
    def __init__(
        self,
        config: ExperimentConfig,
        checkpoint_manager: CheckpointManager,
        fast_model: PolicyValueTransformer | None = None,
        slow_model: PolicyValueTransformer | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(config.runtime.device)
        self.fast = (fast_model or PolicyValueTransformer(config.model)).to(self.device)
        self.slow = (slow_model or copy.deepcopy(self.fast)).to(self.device, dtype=torch.float32)
        self.slow.eval().requires_grad_(False)
        self.state = TrainerState()
        self.checkpoints = checkpoint_manager
        self._hyper_ids = {id(parameter) for parameter in self.fast.hypernet_parameters()}
        self.optimizer = self._build_optimizer()
        if config.runtime.compile_model:
            self.fast.compile(dynamic=False, mode=config.runtime.compile_mode)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        groups: dict[tuple[bool, bool], list[nn.Parameter]] = {}
        for name, parameter in self.fast.named_parameters():
            is_hyper = id(parameter) in self._hyper_ids
            decay = parameter.ndim >= 2 and "norm" not in name and name != "cls_token"
            groups.setdefault((is_hyper, decay), []).append(parameter)
        optimizer_groups: list[dict[str, Any]] = []
        for (is_hyper, decay), parameters in groups.items():
            optimizer_groups.append(
                {
                    "params": parameters,
                    "weight_decay": self.config.training.weight_decay if decay else 0.0,
                    "lr": 0.0,
                    "group_name": "hypernet" if is_hyper else "base",
                }
            )
        return torch.optim.AdamW(
            optimizer_groups,
            lr=0.0,
            betas=(self.config.training.beta1, self.config.training.beta2),
            eps=self.config.training.eps,
            fused=self.device.type == "cuda",
        )

    def _base_lr(self, step: int) -> float:
        warmup = self.config.training.warmup_steps
        if step <= warmup:
            return self.config.training.learning_rate * step / warmup
        return self.config.training.learning_rate * math.sqrt(warmup / step)

    def _set_schedule(self, step: int) -> tuple[float, float]:
        learning_rate = self._base_lr(step)
        hyper = self.config.model.hypernet
        if not hyper.enabled or step <= hyper.freeze_steps:
            scale = 0.0
        else:
            scale = min(1.0, (step - hyper.freeze_steps) / hyper.ramp_steps)
        self.fast.set_hypernet_scale(scale)
        for group in self.optimizer.param_groups:
            if group["group_name"] == "hypernet":
                group["lr"] = (
                    0.0
                    if step <= hyper.freeze_steps
                    else learning_rate * hyper.lr_multiplier
                )
            else:
                group["lr"] = learning_rate
        return learning_rate, scale

    def _tensor_batch(self, batch: SampledBatch) -> tuple[torch.Tensor, ...]:
        arrays = (
            batch.board,
            batch.global_features,
            batch.legal,
            batch.policy,
            batch.value,
            batch.ownership,
            batch.score_margin,
            batch.ownership_mask,
            batch.score_mask,
        )
        tensors = [torch.from_numpy(value) for value in arrays]
        if self.device.type == "cuda":
            tensors = [tensor.pin_memory() for tensor in tensors]
        return tuple(tensor.to(self.device, non_blocking=True) for tensor in tensors)

    def _fault(self, reason: str) -> None:
        self.checkpoints.save_fault(
            self.state.optimizer_step,
            self.checkpoint_payload(),
            reason,
        )
        raise FloatingPointError(reason)

    def train_optimizer_step(
        self,
        sampler: ReplaySampler,
        rng: np.random.Generator,
    ) -> StepMetrics:
        self.fast.train()
        next_step = self.state.optimizer_step + 1
        learning_rate, hyper_scale = self._set_schedule(next_step)
        self.optimizer.zero_grad(set_to_none=True)
        totals = np.zeros(5, dtype=np.float64)
        accumulation = self.config.training.accumulation_steps
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else torch.autocast(device_type="cpu", enabled=False)
        )
        try:
            for _ in range(accumulation):
                batch = sampler.sample_batch(self.config.training.batch_size, rng)
                (
                    board,
                    global_features,
                    legal,
                    policy,
                    value,
                    ownership,
                    score_margin,
                    ownership_mask,
                    score_mask,
                ) = self._tensor_batch(batch)
                targets = TrainingTargets(
                    policy=policy,
                    value=value,
                    ownership=ownership,
                    score_margin=score_margin,
                    ownership_mask=ownership_mask,
                    score_mask=score_mask,
                )
                with autocast:
                    output = self.fast(board, global_features, legal)
                    losses = compute_losses(output, targets, self.config.training)
                if not torch.isfinite(losses.total):
                    self._fault("non-finite training loss")
                (losses.total / accumulation).backward()
                totals += np.asarray(
                    [
                        losses.total.item(),
                        losses.policy.item(),
                        losses.value.item(),
                        losses.ownership.item(),
                        losses.score.item(),
                    ]
                )
        except BaseException:
            self.optimizer.zero_grad(set_to_none=True)
            raise

        hyper_parameters = [
            parameter
            for parameter in self.fast.parameters()
            if id(parameter) in self._hyper_ids and parameter.grad is not None
        ]
        if hyper_parameters:
            hyper_norm = torch.nn.utils.clip_grad_norm_(
                hyper_parameters,
                self.config.model.hypernet.grad_clip,
            )
            if not torch.isfinite(hyper_norm):
                self._fault("non-finite hypernetwork gradient")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.fast.parameters(),
            self.config.training.gradient_clip,
        )
        if not torch.isfinite(gradient_norm):
            self._fault("non-finite model gradient")
        self.optimizer.step()
        if any(not torch.isfinite(parameter).all() for parameter in self.fast.parameters()):
            self._fault("non-finite model parameter after optimizer step")

        effective_batch = self.config.training.batch_size * accumulation
        self.state.optimizer_step = next_step
        self.state.samples_seen += effective_batch
        self.state.ema_pending_samples += effective_batch
        if next_step % self.config.training.ema_update_interval == 0:
            update_slow_weights(
                self.slow,
                self.fast,
                self.state.ema_pending_samples,
                self.config.training.ema_half_life_samples,
            )
            self.state.ema_pending_samples = 0
        totals /= accumulation
        return StepMetrics(
            step=next_step,
            learning_rate=learning_rate,
            hypernet_scale=hyper_scale,
            total_loss=float(totals[0]),
            policy_loss=float(totals[1]),
            value_loss=float(totals[2]),
            ownership_loss=float(totals[3]),
            score_loss=float(totals[4]),
            gradient_norm=float(gradient_norm),
        )

    def train_steps(
        self,
        count: int,
        sampler: ReplaySampler,
        rng: np.random.Generator,
    ) -> list[StepMetrics]:
        return [self.train_optimizer_step(sampler, rng) for _ in range(count)]

    def checkpoint_payload(self, rng: np.random.Generator | None = None) -> dict[str, Any]:
        payload = {
            **checkpoint_metadata(self.config.canonical_json(), self.config.sha256),
            "trainer_state": asdict(self.state),
            "fast_state": self.fast.state_dict(),
            "slow_state": self.slow.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng_state": None if rng is None else rng.bit_generator.state,
        }
        return payload

    def save_checkpoint(
        self,
        rng: np.random.Generator | None = None,
        replay_metadata: dict[str, Any] | None = None,
    ):
        payload = self.checkpoint_payload(rng)
        payload["replay_metadata"] = replay_metadata
        return self.checkpoints.save_full(
            self.state.optimizer_step,
            payload,
        )

    def publish(self):
        self.state.last_published_step = self.state.optimizer_step
        metadata = checkpoint_metadata(self.config.canonical_json(), self.config.sha256)
        return self.checkpoints.save_publication(
            self.state.optimizer_step,
            self.slow.state_dict(),
            metadata,
        )

    def restore(self, path, rng: np.random.Generator | None = None) -> None:
        payload = self.checkpoints.load(path, map_location=self.device)
        if payload["config_sha256"] != self.config.sha256:
            raise ValueError("checkpoint configuration does not match this run")
        self.fast.load_state_dict(payload["fast_state"])
        self.slow.load_state_dict(payload["slow_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.state = TrainerState(**payload["trainer_state"])
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        if torch.cuda.is_available() and payload["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
        if rng is not None and payload["numpy_rng_state"] is not None:
            rng.bit_generator.state = payload["numpy_rng_state"]
