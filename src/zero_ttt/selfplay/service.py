"""Application service for publication-backed self-play collection."""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

import torch

from zero_ttt._io import canonical_json_bytes, sha256_bytes
from zero_ttt.config import ExperimentConfig
from zero_ttt.data.ingestion import DEFAULT_TARGET_SHARD_BYTES
from zero_ttt.game.features import FEATURE_SCHEMA_ID
from zero_ttt.game.rules import RULES_ID
from zero_ttt.inference import BatchedInferenceBroker, PublicationPositionEvaluator
from zero_ttt.selfplay.collector import CollectionSummary, SelfPlayCollector, search_config_sha256


class SelfPlayService:
    """Own evaluator identity, broker lifetime, and collector construction."""

    def __init__(
        self,
        config: ExperimentConfig,
        publication: str | Path,
        *,
        store_root: str | Path,
        catalog_path: str | Path,
    ) -> None:
        self.config = config
        self.store_root = Path(store_root)
        self.catalog_path = Path(catalog_path)
        self.evaluator = PublicationPositionEvaluator(
            publication,
            device=config.runtime.device,
            inference_batch_size=config.selfplay.inference_batch_size,
            compile_model=config.selfplay.compile_inference,
            compile_mode=config.execution.compile_mode,
        )
        search_hash = search_config_sha256(config)
        self.evaluator_id = sha256_bytes(
            canonical_json_bytes(
                [
                    self.evaluator.model_version,
                    FEATURE_SCHEMA_ID,
                    RULES_ID,
                    search_hash,
                ]
            )
        )
        self._broker: BatchedInferenceBroker | None = None

    def __enter__(self) -> SelfPlayService:
        if self._broker is not None:
            raise RuntimeError("self-play service is already open")
        if self.evaluator.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.evaluator.device)
        broker = BatchedInferenceBroker(
            self.evaluator,
            batch_size=self.config.selfplay.inference_batch_size,
            batch_wait_ms=self.config.selfplay.batch_wait_ms,
            cache_size=self.config.selfplay.inference_cache_size,
        )
        self._broker = broker.__enter__()
        return self

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        broker, self._broker = self._broker, None
        if broker is not None:
            broker.__exit__(error_type, error, traceback)

    def collect(
        self,
        *,
        games: int,
        seed: int,
        target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
    ) -> CollectionSummary:
        if self._broker is None:
            raise RuntimeError("self-play service must be opened before collection")
        return SelfPlayCollector(
            self.config,
            self._broker,
            publication_sha256=self.evaluator.publication_sha256,
            evaluator_id=self.evaluator_id,
            store_root=self.store_root,
            catalog_path=self.catalog_path,
            games=games,
            seed=seed,
            target_shard_bytes=target_shard_bytes,
        ).collect()

    def gpu_peak_allocated_bytes(self) -> int:
        if self.evaluator.device.type != "cuda":
            return 0
        torch.cuda.synchronize(self.evaluator.device)
        return int(torch.cuda.max_memory_allocated(self.evaluator.device))
