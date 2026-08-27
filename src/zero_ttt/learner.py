"""The single offline learner implementation for normalized training batches."""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from zero_ttt.config import ExperimentConfig
from zero_ttt.data.contracts import BatchSource, TrainBatch
from zero_ttt.model import BasePolicyValueModel, PolicyValueTransformer
from zero_ttt.training.checkpoint import CheckpointManager, checkpoint_metadata
from zero_ttt.training.contracts import CheckpointSummary, LearnerDataIdentity
from zero_ttt.training.ema import update_slow_weights
from zero_ttt.training.gradients import (
    NonFiniteGradientError,
    clip_model_gradients,
    parameters_are_finite,
)
from zero_ttt.training.losses import TrainingTargets, compute_losses


@dataclass(slots=True)
class LearnerState:
    optimizer_step: int = 0
    samples_seen: int = 0
    ema_pending_samples: int = 0
    next_ema_sample: int = 0
    next_publish_sample: int = 0
    last_published_step: int = 0
    last_published_samples: int = 0
    run_id: str = ""


@dataclass(frozen=True, slots=True)
class StepMetrics:
    step: int
    learning_rate: float
    total_loss: float
    policy_loss: float
    value_loss: float
    ownership_loss: float
    score_loss: float
    base_gradient_norm: float
    hypernet_gradient_norm: float | None
    hyper_a_saturation: float
    hyper_b_saturation: float
    hyper_dynamic_rms: float
    hyper_static_rms: float
    ema_update_seconds: float | None


def _next_boundary(samples_seen: int, interval: int) -> int:
    return (samples_seen // interval + 1) * interval


class Learner:
    def __init__(
        self,
        config: ExperimentConfig,
        checkpoint_manager: CheckpointManager,
        fast_model: BasePolicyValueModel | None = None,
        slow_model: BasePolicyValueModel | None = None,
        data_identity: LearnerDataIdentity | None = None,
        run_id: str | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(config.runtime.device)
        self.ema_device = torch.device(config.runtime.ema_device)
        self.fast = (
            fast_model or PolicyValueTransformer(config.model, config.execution)
        ).to(self.device)
        self.slow = (
            slow_model or PolicyValueTransformer(config.model, config.execution)
        ).to(self.ema_device, dtype=torch.float32)
        self.slow.load_state_dict(self.fast.state_dict())
        self.slow.eval().requires_grad_(False)
        self.fast.configure_execution(config.execution)
        self.slow.configure_execution(config.execution)
        training = config.training
        self.state = LearnerState(
            next_ema_sample=training.ema_update_interval_samples,
            next_publish_sample=training.publish_interval_samples,
            run_id=run_id or uuid.uuid4().hex,
        )
        self.data_identity = data_identity
        self.checkpoints = checkpoint_manager
        self._parameter_groups = self.fast.parameter_groups()
        self.optimizer = self._build_optimizer()
        if config.execution.compile_model:
            self.fast.compile_training_components(
                dynamic=False,
                mode=config.execution.compile_mode,
            )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        optimizer_groups: list[dict[str, Any]] = []
        for group in self._parameter_groups:
            for parameters, weight_decay in (
                (group.decay, self.config.training.weight_decay),
                (group.no_decay, 0.0),
            ):
                if parameters:
                    optimizer_groups.append(
                        {
                            "params": parameters,
                            "weight_decay": weight_decay,
                            "lr": 0.0,
                            "group_name": group.name,
                        }
                    )
        return torch.optim.AdamW(
            optimizer_groups,
            lr=0.0,
            betas=(self.config.training.beta1, self.config.training.beta2),
            eps=self.config.training.eps,
            fused=self.device.type == "cuda",
        )

    def _base_lr(self, samples_seen: int) -> float:
        if samples_seen <= 0:
            return 0.0
        warmup = self.config.training.warmup_samples
        if samples_seen <= warmup:
            return self.config.training.learning_rate * samples_seen / warmup
        return self.config.training.learning_rate * math.sqrt(warmup / samples_seen)

    def _set_schedule(self, samples_seen: int) -> float:
        learning_rate = self._base_lr(samples_seen)
        hyper = self.config.training.hypernet
        for group in self.optimizer.param_groups:
            if group["group_name"] == "hypernet":
                group["lr"] = learning_rate * hyper.learning_rate_multiplier
            else:
                group["lr"] = learning_rate
        return learning_rate

    def _tensor_batch(self, batch: TrainBatch) -> tuple[torch.Tensor, ...]:
        if batch.board.shape[0] != self.config.training.batch_size:
            raise ValueError(
                f"BatchSource returned {batch.board.shape[0]} samples; "
                f"expected {self.config.training.batch_size}"
            )
        arrays = (
            batch.board,
            batch.global_features,
            batch.legal,
            batch.policy,
            batch.value,
            batch.ownership,
            batch.score_margin,
            batch.value_mask,
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
        source: BatchSource,
        rng: np.random.Generator,
    ) -> StepMetrics:
        self.fast.train()
        next_step = self.state.optimizer_step + 1
        effective_batch = self.config.training.effective_batch_size
        next_samples = self.state.samples_seen + effective_batch
        learning_rate = self._set_schedule(next_samples)
        self.optimizer.zero_grad(set_to_none=False)
        totals = np.zeros(9, dtype=np.float64)
        accumulation = self.config.training.accumulation_steps
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else torch.autocast(device_type="cpu", enabled=False)
        )
        try:
            for _ in range(accumulation):
                batch = source.next_batch(self.config.training.batch_size, rng)
                (
                    board,
                    global_features,
                    legal,
                    policy,
                    value,
                    ownership,
                    score_margin,
                    value_mask,
                    ownership_mask,
                    score_mask,
                ) = self._tensor_batch(batch)
                targets = TrainingTargets(
                    policy=policy,
                    value=value,
                    ownership=ownership,
                    score_margin=score_margin,
                    value_mask=value_mask,
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
                        output.diagnostics.hyper_a_saturation.item(),
                        output.diagnostics.hyper_b_saturation.item(),
                        output.diagnostics.hyper_dynamic_rms.item(),
                        output.diagnostics.hyper_static_rms.item(),
                    ]
                )
        except BaseException:
            self.optimizer.zero_grad(set_to_none=True)
            raise

        try:
            gradient_norms = clip_model_gradients(
                self._parameter_groups,
                base_max_norm=self.config.training.gradient_clip,
                hypernet_max_norm=self.config.training.hypernet.gradient_clip,
            )
        except NonFiniteGradientError as error:
            self._fault(str(error))
        self.optimizer.step()
        if not parameters_are_finite(self.fast.parameters()):
            self._fault("non-finite model parameter after optimizer step")

        self.state.optimizer_step = next_step
        self.state.samples_seen = next_samples
        self.state.ema_pending_samples += effective_batch
        ema_update_seconds: float | None = None
        if self.state.samples_seen >= self.state.next_ema_sample:
            ema_started = time.perf_counter()
            update_slow_weights(
                self.slow,
                self.fast,
                self.state.ema_pending_samples,
                self.config.training.ema_half_life_samples,
            )
            ema_update_seconds = time.perf_counter() - ema_started
            self.state.ema_pending_samples = 0
            self.state.next_ema_sample = _next_boundary(
                self.state.samples_seen,
                self.config.training.ema_update_interval_samples,
            )
        totals /= accumulation
        return StepMetrics(
            step=next_step,
            learning_rate=learning_rate,
            total_loss=float(totals[0]),
            policy_loss=float(totals[1]),
            value_loss=float(totals[2]),
            ownership_loss=float(totals[3]),
            score_loss=float(totals[4]),
            base_gradient_norm=gradient_norms.base,
            hypernet_gradient_norm=gradient_norms.hypernet,
            hyper_a_saturation=float(totals[5]),
            hyper_b_saturation=float(totals[6]),
            hyper_dynamic_rms=float(totals[7]),
            hyper_static_rms=float(totals[8]),
            ema_update_seconds=ema_update_seconds,
        )

    def train_steps(
        self,
        count: int,
        source: BatchSource,
        rng: np.random.Generator,
    ) -> list[StepMetrics]:
        if count < 0:
            raise ValueError("count cannot be negative")
        return [self.train_optimizer_step(source, rng) for _ in range(count)]

    @property
    def publication_due(self) -> bool:
        return self.state.samples_seen >= self.state.next_publish_sample

    def checkpoint_payload(
        self, rng: np.random.Generator | None = None
    ) -> dict[str, Any]:
        return {
            **checkpoint_metadata(self.config.canonical_json(), self.config.sha256),
            "learner_state": asdict(self.state),
            "data_identity": None
            if self.data_identity is None
            else asdict(self.data_identity),
            "fast_state": self.fast.state_dict(),
            "slow_state": self.slow.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
            "numpy_rng_state": None if rng is None else rng.bit_generator.state,
        }

    def save_checkpoint(self, rng: np.random.Generator | None = None):
        return self.checkpoints.save_full(
            self.state.optimizer_step,
            self.checkpoint_payload(rng),
        )

    def publish(self):
        metadata = checkpoint_metadata(self.config.canonical_json(), self.config.sha256)
        publication = self.checkpoints.save_publication(
            self.state.run_id,
            self.state.optimizer_step,
            self.state.samples_seen,
            self.slow.state_dict(),
            metadata,
        )
        self.state.last_published_step = self.state.optimizer_step
        self.state.last_published_samples = self.state.samples_seen
        self.state.next_publish_sample = _next_boundary(
            self.state.samples_seen,
            self.config.training.publish_interval_samples,
        )
        return publication

    def publish_if_due(self):
        return self.publish() if self.publication_due else None

    def _restore_payload(
        self,
        path,
        rng: np.random.Generator | None,
        *,
        allow_data_transition: bool,
    ) -> LearnerDataIdentity | None:
        payload = self.checkpoints.load(path, map_location="cpu")
        if (
            payload.get("config_sha256") != self.config.sha256
            or payload.get("config_json") != self.config.canonical_json()
        ):
            raise ValueError("checkpoint configuration does not match this run")
        summary = CheckpointSummary.from_payload(payload)
        stored_identity = summary.data_identity
        expected_identity = self.data_identity
        if not allow_data_transition and stored_identity != expected_identity:
            raise ValueError(
                "checkpoint data snapshot or sampling configuration does not match"
            )
        if allow_data_transition:
            if stored_identity is None or expected_identity is None:
                raise ValueError("data transition requires old and new data identities")
            if stored_identity == expected_identity:
                raise ValueError("data transition requires a different data identity")
        state = summary.learner_state
        self.fast.load_state_dict(payload["fast_state"])
        self.slow.load_state_dict(payload["slow_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.state = LearnerState(**state)
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        if torch.cuda.is_available() and payload["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
        if rng is not None and payload["numpy_rng_state"] is not None:
            rng.bit_generator.state = payload["numpy_rng_state"]
        return stored_identity

    def restore(self, path, rng: np.random.Generator | None = None) -> None:
        self._restore_payload(path, rng, allow_data_transition=False)

    def restore_for_data_transition(
        self,
        path,
        rng: np.random.Generator | None = None,
    ) -> LearnerDataIdentity:
        """Restore all training state while explicitly adopting a new data identity.

        The learner must be constructed with the destination ``data_identity``. This
        deliberately keeps normal ``restore`` strict while allowing an orchestrator
        to roll an immutable training snapshot without resetting optimizer, EMA, RNG,
        or sample-based schedules.
        """

        previous = self._restore_payload(path, rng, allow_data_transition=True)
        if previous is None:
            raise ValueError("data transition checkpoint has no previous data identity")
        return previous
