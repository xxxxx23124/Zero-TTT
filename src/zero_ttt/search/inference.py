"""Batched model evaluation and a dedicated inference queue."""

from __future__ import annotations

import contextlib
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from zero_ttt.config import RuntimeConfig, SearchConfig
from zero_ttt.game.features import encode_position
from zero_ttt.game.state import GameState
from zero_ttt.model.transformer import PolicyValueTransformer
from zero_ttt.search.cache import EvaluationCache
from zero_ttt.search.protocol import BatchEvaluator, Evaluation


class TorchBatchEvaluator(BatchEvaluator):
    def __init__(
        self,
        model: PolicyValueTransformer,
        runtime: RuntimeConfig,
        max_batch_size: int,
        model_version: int,
    ) -> None:
        self.device = torch.device(runtime.device)
        self.model_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = model.to(self.device, dtype=self.model_dtype).eval().requires_grad_(False)
        self.max_batch_size = max_batch_size
        self.model_version = model_version
        if runtime.compile_model:
            self.model.compile(dynamic=False, mode=runtime.compile_mode)

    def load_publication(
        self,
        slow_state: dict[str, torch.Tensor],
        model_version: int,
    ) -> None:
        self.model.load_state_dict(slow_state)
        self.model.eval().requires_grad_(False)
        self.model_version = model_version

    @staticmethod
    def _bucket_size(size: int, maximum: int) -> int:
        for bucket in (1, 2, 4, 8, 16):
            if size <= bucket <= maximum:
                return bucket
        return size

    def evaluate_batch(
        self,
        states: Sequence[GameState],
        model_version: int,
    ) -> list[Evaluation]:
        if model_version != self.model_version:
            raise ValueError(
                f"requested model_version={model_version}, "
                f"but evaluator has model_version={self.model_version}"
            )
        if not states or len(states) > self.max_batch_size:
            raise ValueError("invalid inference batch size")
        encoded = [encode_position(state) for state in states]
        bucket = self._bucket_size(len(encoded), self.max_batch_size)
        while len(encoded) < bucket:
            encoded.append(encoded[-1])
        host_tensors = [
            torch.from_numpy(np.stack([item.board for item in encoded])),
            torch.from_numpy(np.stack([item.global_features for item in encoded])),
            torch.from_numpy(np.stack([item.legal for item in encoded])),
        ]
        if self.device.type == "cuda":
            host_tensors = [tensor.pin_memory() for tensor in host_tensors]
        board, global_features, legal = (
            tensor.to(self.device, non_blocking=True) for tensor in host_tensors
        )
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), autocast:
            output = self.model(board, global_features, legal)
            policies = torch.softmax(output.policy_logits.float(), dim=-1).cpu().numpy()
            values = output.value.float().squeeze(-1).cpu().numpy()
            ownership = output.ownership.float().cpu().numpy()
            scores = output.score_margin.float().squeeze(-1).cpu().numpy()
        return [
            Evaluation(
                policy=policies[index],
                value=float(values[index]),
                ownership=ownership[index],
                score_margin=float(scores[index]),
            )
            for index in range(len(states))
        ]


@dataclass(slots=True)
class InferenceRequest:
    state: GameState
    model_version: int
    future: Future[Evaluation]


class InferenceServer:
    """Expose a blocking leaf evaluator while batching work on one thread."""

    def __init__(
        self,
        backend: BatchEvaluator,
        config: SearchConfig,
        cache: EvaluationCache | None = None,
    ) -> None:
        self.backend = backend
        self.config = config
        self.cache = cache or EvaluationCache(config.inference_cache_size)
        self._queue: queue.Queue[InferenceRequest | None] = queue.Queue()
        self._lifecycle_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="zero-ttt-inference", daemon=True)
        self._closed = False
        self._thread.start()

    def evaluate(self, state: GameState, model_version: int) -> Evaluation:
        cached = self.cache.get(state, model_version)
        if cached is not None:
            return cached
        future: Future[Evaluation] = Future()
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("inference server is closed")
            self._queue.put(
                InferenceRequest(state=state, model_version=model_version, future=future)
            )
        return future.result()

    def _run(self) -> None:
        deferred: list[InferenceRequest] = []
        while True:
            if deferred:
                first = deferred.pop(0)
            else:
                first = self._queue.get()
                if first is None:
                    return
            batch = [first]
            deadline = time.monotonic() + self.config.batch_delay_ms / 1000.0
            while len(batch) < self.config.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    request = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if request is None:
                    self._queue.put(None)
                    break
                if request.model_version == first.model_version:
                    batch.append(request)
                else:
                    deferred.append(request)

            groups: dict[tuple[object, ...], list[InferenceRequest]] = {}
            for request in batch:
                cached = self.cache.get(request.state, request.model_version)
                if cached is not None:
                    request.future.set_result(cached)
                    continue
                groups.setdefault(request.state.identity(), []).append(request)
            if not groups:
                continue
            unique = [requests[0] for requests in groups.values()]
            try:
                evaluations = self.backend.evaluate_batch(
                    [request.state for request in unique],
                    first.model_version,
                )
                if len(evaluations) != len(unique):
                    raise RuntimeError("batch evaluator returned the wrong number of results")
                for request, evaluation in zip(unique, evaluations, strict=True):
                    self.cache.put(request.state, request.model_version, evaluation)
                    for duplicate in groups[request.state.identity()]:
                        duplicate.future.set_result(evaluation)
            except BaseException as error:
                for requests in groups.values():
                    for request in requests:
                        if not request.future.done():
                            request.future.set_exception(error)

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            self._queue.put(None)
        self._thread.join()
        while True:
            try:
                request = self._queue.get_nowait()
            except queue.Empty:
                break
            if request is not None and not request.future.done():
                request.future.cancel()

    def __enter__(self) -> "InferenceServer":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
