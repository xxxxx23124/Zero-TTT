"""Read-only FP32 policy-value inference from an immutable publication."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from zero_ttt._io import sha256_file
from zero_ttt.config import model_config_from_mapping
from zero_ttt.model import PolicyValueTransformer
from zero_ttt.model.transformer import eager_execution_config
from zero_ttt.precision import (
    TENSOR_PRECISION,
    configure_strict_fp32,
    require_fp32_module,
    require_fp32_state,
)
from zero_ttt.versioning import MODEL_ARTIFACT_SCHEMA

from zero_ttt_selfplay_worker.inference.contracts import InferenceBatch, InferenceOutput


class PublicationPositionEvaluator:
    """A fixed-shape evaluator backed by one immutable publication."""

    def __init__(
        self,
        publication_path: str | Path,
        *,
        device: str | torch.device,
        inference_batch_size: int = 16,
        compile_model: bool = False,
        compile_mode: str = "default",
    ) -> None:
        configure_strict_fp32()
        if inference_batch_size <= 0:
            raise ValueError("inference_batch_size must be positive")
        self.publication_path = Path(publication_path)
        self.publication_sha256 = sha256_file(self.publication_path)
        payload = torch.load(self.publication_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("publication payload must be a mapping")
        MODEL_ARTIFACT_SCHEMA.require(payload.get("checkpoint_schema_version"))
        if payload.get("tensor_precision") != TENSOR_PRECISION:
            raise ValueError(f"publication tensor precision must be {TENSOR_PRECISION}")
        slow_state = payload.get("slow_state")
        if not isinstance(slow_state, dict):
            raise ValueError("publication is missing slow weights")
        require_fp32_state(slow_state, "publication slow_state")
        try:
            config_mapping = json.loads(payload["config_json"])
            model_config = model_config_from_mapping(config_mapping["model"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("publication has no valid embedded model configuration") from error

        self.device = torch.device(device)
        self.inference_batch_size = inference_batch_size
        model = PolicyValueTransformer(model_config, eager_execution_config())
        model = model.to(device=self.device).eval().requires_grad_(False)
        incompatible = model.load_state_dict(payload["slow_state"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError("publication state does not match its model configuration")
        require_fp32_module(model, "publication model")
        self._model = (
            torch.compile(model, dynamic=False, mode=compile_mode) if compile_model else model
        )
        self._model_version = (
            f"{payload.get('run_id', 'unknown')}:{int(payload['model_version'])}:"
            f"{self.publication_sha256}"
        )

    @property
    def model_version(self) -> str:
        return self._model_version

    def evaluate(self, batch: InferenceBatch) -> InferenceOutput:
        real_size = batch.board.shape[0]
        if real_size > self.inference_batch_size:
            raise ValueError("inference batch exceeds the fixed publication batch size")

        board = batch.board
        global_features = batch.global_features
        legal = batch.legal
        if real_size < self.inference_batch_size:
            padding = self.inference_batch_size - real_size
            board = torch.cat((board, board[-1:].expand(padding, -1, -1, -1)), dim=0)
            global_features = torch.cat(
                (global_features, global_features[-1:].expand(padding, -1)), dim=0
            )
            legal = torch.cat((legal, legal[-1:].expand(padding, -1)), dim=0)

        board = board.to(self.device, non_blocking=self.device.type == "cuda")
        global_features = global_features.to(self.device, non_blocking=self.device.type == "cuda")
        legal = legal.to(self.device, non_blocking=self.device.type == "cuda")
        with torch.inference_mode():
            output = self._model(board, global_features, legal)
        ownership = output.ownership[:real_size]
        score_margin = output.score_margin[:real_size].reshape(real_size)
        return InferenceOutput(
            policy_logits=output.policy_logits[:real_size],
            value=output.value[:real_size].reshape(real_size),
            ownership=ownership,
            score_margin=score_margin,
        )
