"""Strict, file-only experiment configuration."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import UnionType
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

from zero_ttt.versioning import EXPERIMENT_CONFIG_SCHEMA


@dataclass(frozen=True, slots=True)
class GameConfig:
    board_size: int
    komi_half_points: int
    max_moves: int
    history_length: int


@dataclass(frozen=True, slots=True)
class RoPEConfig:
    base: float
    scale: float
    rotary_dim: int
    centered: bool
    learnable: bool


@dataclass(frozen=True, slots=True)
class HypernetConfig:
    enabled: bool
    num_layers: int
    rank: int
    hidden_dim: int
    context_gradient_scale: float


@dataclass(frozen=True, slots=True)
class DepthMixingConfig:
    enabled: bool
    dilation: int
    period: int


@dataclass(frozen=True, slots=True)
class ModelConfig:
    input_planes: int
    global_features: int
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    rope: RoPEConfig
    hypernet: HypernetConfig
    depth_mixing: DepthMixingConfig


@dataclass(frozen=True, slots=True)
class HypernetTrainingConfig:
    learning_rate_multiplier: float
    gradient_clip: float


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    batch_size: int
    accumulation_steps: int
    learning_rate: float
    beta1: float
    beta2: float
    eps: float
    weight_decay: float
    warmup_samples: int
    gradient_clip: float
    policy_loss_weight: float
    value_loss_weight: float
    ownership_loss_weight: float
    score_loss_weight: float
    ema_half_life_samples: int
    ema_update_interval_samples: int
    publish_interval_samples: int
    checkpoint_keep: int
    hypernet: HypernetTrainingConfig

    @property
    def effective_batch_size(self) -> int:
        """Training positions consumed by one optimizer update."""

        return self.batch_size * self.accumulation_steps


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    activation_checkpoint: bool
    activation_checkpoint_stride: int
    compile_model: bool
    compile_mode: str


@dataclass(frozen=True, slots=True)
class SearchConfig:
    max_simulations: int
    uct_c: float
    dirichlet_epsilon: float
    dirichlet_alpha: float
    temperature: float
    temperature_drop_ply: int


@dataclass(frozen=True, slots=True)
class SelfPlayConfig:
    actor_count: int
    inference_batch_size: int
    batch_wait_ms: float
    inference_cache_size: int
    compile_inference: bool


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    run_dir: str
    device: str
    ema_device: str


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    schema_version: int
    run_name: str
    seed: int
    game: GameConfig
    model: ModelConfig
    training: TrainingConfig
    execution: ExecutionConfig
    search: SearchConfig
    selfplay: SelfPlayConfig
    runtime: RuntimeConfig

    def canonical_json(self) -> str:
        return json.dumps(
            dataclasses.asdict(self),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def run_dir(self) -> Path:
        return Path(self.runtime.run_dir)


T = TypeVar("T")


def _coerce_value(expected: Any, value: Any, path: str) -> Any:
    origin = get_origin(expected)
    if origin in (Union, UnionType):
        for candidate in get_args(expected):
            try:
                return _coerce_value(candidate, value, path)
            except (TypeError, ValueError):
                continue
        raise TypeError(f"{path}: value does not match {expected!r}")
    if dataclasses.is_dataclass(expected):
        if not isinstance(value, dict):
            raise TypeError(f"{path}: expected table")
        return _construct_dataclass(expected, value, path)
    if expected is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{path}: expected float")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{path}: expected finite float")
        return result
    if expected is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{path}: expected int")
        return value
    if expected is bool:
        if not isinstance(value, bool):
            raise TypeError(f"{path}: expected bool")
        return value
    if expected is str:
        if not isinstance(value, str):
            raise TypeError(f"{path}: expected string")
        return value
    raise TypeError(f"{path}: unsupported configuration type {expected!r}")


def _construct_dataclass(cls: type[T], data: dict[str, Any], path: str) -> T:
    fields = {field.name: field for field in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(fields))
    missing = sorted(set(fields) - set(data))
    if unknown:
        raise ValueError(f"{path}: unknown fields: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"{path}: missing fields: {', '.join(missing)}")
    hints = get_type_hints(cls)
    kwargs = {
        name: _coerce_value(hints[name], data[name], f"{path}.{name}")
        for name in fields
    }
    return cls(**kwargs)


def validate_config(config: ExperimentConfig) -> None:
    EXPERIMENT_CONFIG_SCHEMA.require(config.schema_version)
    if config.game.board_size != 19:
        raise ValueError("only 19x19 is supported")
    if config.game.history_length != 8:
        raise ValueError("the feature schema requires history_length=8")
    if config.game.max_moves < 2:
        raise ValueError("game.max_moves must be at least 2")
    model = config.model
    if model.input_planes != 25 or model.global_features != 5:
        raise ValueError("the current feature schema requires 25 point planes and 5 global features")
    if model.d_model <= 0 or model.n_heads <= 0 or model.d_model % model.n_heads:
        raise ValueError("model.d_model must be positive and divisible by model.n_heads")
    if model.n_layers <= 0 or model.d_ff <= 0:
        raise ValueError("model layer counts and widths must be positive")
    head_dim = model.d_model // model.n_heads
    rope = model.rope
    if rope.rotary_dim <= 0 or rope.rotary_dim > head_dim or rope.rotary_dim % 4:
        raise ValueError("rope.rotary_dim must be positive, <= head_dim, and divisible by 4")
    if rope.base <= 1.0 or rope.scale <= 0.0:
        raise ValueError("RoPE base must be >1 and scale must be positive")
    hyper = model.hypernet
    if not (0 <= hyper.num_layers <= model.n_layers):
        raise ValueError("hypernet.num_layers must be within model depth")
    if hyper.rank <= 0 or hyper.hidden_dim <= 0:
        raise ValueError("hypernet rank and hidden_dim must be positive")
    for name, value in (
        ("context_gradient_scale", hyper.context_gradient_scale),
    ):
        if value < 0:
            raise ValueError(f"hypernet.{name} must be non-negative")
    depth_mixing = model.depth_mixing
    if depth_mixing.dilation <= 0 or depth_mixing.period <= 0:
        raise ValueError("depth_mixing dilation and period must be positive")
    if depth_mixing.dilation > model.n_layers or depth_mixing.period > model.n_layers:
        raise ValueError("depth_mixing dilation and period cannot exceed model depth")
    train = config.training
    positive_train = (
        train.batch_size,
        train.accumulation_steps,
        train.learning_rate,
        train.eps,
        train.warmup_samples,
        train.gradient_clip,
        train.ema_half_life_samples,
        train.ema_update_interval_samples,
        train.publish_interval_samples,
        train.checkpoint_keep,
    )
    if any(value <= 0 for value in positive_train):
        raise ValueError("training sizes, rates, intervals, and limits must be positive")
    if train.weight_decay < 0:
        raise ValueError("training.weight_decay must be non-negative")
    if train.hypernet.learning_rate_multiplier < 0:
        raise ValueError("training.hypernet.learning_rate_multiplier must be non-negative")
    if train.hypernet.gradient_clip <= 0:
        raise ValueError("training.hypernet.gradient_clip must be positive")
    if not (0 < train.beta1 < 1 and 0 < train.beta2 < 1):
        raise ValueError("AdamW beta values must be in (0, 1)")
    execution = config.execution
    if execution.activation_checkpoint_stride <= 0:
        raise ValueError("execution.activation_checkpoint_stride must be positive")
    if not execution.compile_mode:
        raise ValueError("execution.compile_mode cannot be empty")
    search = config.search
    if search.max_simulations < 2:
        raise ValueError("search.max_simulations must be at least 2")
    if search.uct_c <= 0 or search.dirichlet_alpha <= 0:
        raise ValueError("search constants must be positive")
    if not 0 <= search.dirichlet_epsilon <= 1:
        raise ValueError("search.dirichlet_epsilon must be in [0, 1]")
    if search.temperature < 0 or search.temperature_drop_ply < 0:
        raise ValueError("search temperature settings must be non-negative")
    selfplay = config.selfplay
    if (
        selfplay.actor_count <= 0
        or selfplay.inference_batch_size <= 0
        or selfplay.inference_cache_size <= 0
        or selfplay.batch_wait_ms < 0
    ):
        raise ValueError("selfplay batching settings are invalid")
    if selfplay.actor_count < selfplay.inference_batch_size:
        raise ValueError("selfplay.actor_count must cover one inference batch")
    if config.runtime.device not in {"cpu", "cuda"}:
        raise ValueError("runtime.device must be 'cpu' or 'cuda'")
    if config.runtime.ema_device not in {"cpu", "cuda"}:
        raise ValueError("runtime.ema_device must be 'cpu' or 'cuda'")


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    config = config_from_mapping(raw)
    return config


def config_from_mapping(raw: dict[str, Any]) -> ExperimentConfig:
    """Build an exact current-schema configuration."""

    EXPERIMENT_CONFIG_SCHEMA.require(raw.get("schema_version"))
    config = _construct_dataclass(
        ExperimentConfig,
        raw,
        "config",
    )
    validate_config(config)
    return config


def model_config_from_mapping(raw: dict[str, Any]) -> ModelConfig:
    """Parse only the stable model section embedded in a publication."""

    return _construct_dataclass(ModelConfig, raw, "config.model")
