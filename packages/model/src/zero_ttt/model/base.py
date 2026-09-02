"""Template base class for policy-value models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import final

import torch
from torch import nn
from zero_ttt.config import ExecutionConfig
from zero_ttt.model.contracts import (
    BackboneOutput,
    ModelOutput,
    ModelParameterGroup,
    ModelPredictions,
)


class BasePolicyValueModel(nn.Module, ABC):
    """Own the stable macro forward while subclasses supply model mathematics."""

    def __init__(
        self,
        *,
        input_planes: int,
        global_features: int,
        board_size: int,
        action_size: int,
    ) -> None:
        super().__init__()
        self.input_planes = input_planes
        self.global_features = global_features
        self.board_size = board_size
        self.action_size = action_size

    def _validate_inputs(
        self,
        board_features: torch.Tensor,
        global_features: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> None:
        if board_features.ndim != 4 or board_features.shape[1:] != (
            self.input_planes,
            self.board_size,
            self.board_size,
        ):
            raise ValueError("board_features has the wrong shape")
        batch = board_features.shape[0]
        if batch <= 0:
            raise ValueError("model batches cannot be empty")
        if global_features.shape != (batch, self.global_features):
            raise ValueError("global_features has the wrong shape")
        if legal_mask.shape != (batch, self.action_size):
            raise ValueError("legal_mask has the wrong shape")
        if board_features.dtype != torch.float32 or global_features.dtype != torch.float32:
            raise TypeError("board_features and global_features must have torch.float32 dtype")
        if legal_mask.dtype != torch.bool:
            raise TypeError("legal_mask must have boolean dtype")
        if not (board_features.device == global_features.device == legal_mask.device):
            raise ValueError("all model inputs must be on the same device")

    @final
    def forward(
        self,
        board_features: torch.Tensor,
        global_features: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> ModelOutput:
        self._validate_inputs(board_features, global_features, legal_mask)
        tokens = self._encode_tokens(board_features, global_features)
        backbone = self._forward_backbone(tokens)
        predictions = self._decode_outputs(backbone.hidden)
        policy_logits = predictions.policy_logits.masked_fill(~legal_mask, -torch.inf)
        return ModelOutput(
            policy_logits=policy_logits,
            value=predictions.value,
            ownership=predictions.ownership,
            score_margin=predictions.score_margin,
            diagnostics=backbone.diagnostics,
        )

    @abstractmethod
    def _encode_tokens(
        self,
        board_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _forward_backbone(self, hidden: torch.Tensor) -> BackboneOutput:
        raise NotImplementedError

    @abstractmethod
    def _decode_outputs(self, hidden: torch.Tensor) -> ModelPredictions:
        raise NotImplementedError

    @abstractmethod
    def _parameter_group_members(self) -> Mapping[str, Iterable[nn.Parameter]]:
        raise NotImplementedError

    def _no_weight_decay_parameters(self) -> Iterable[nn.Parameter]:
        return ()

    @abstractmethod
    def configure_execution(self, config: ExecutionConfig) -> None:
        raise NotImplementedError

    @abstractmethod
    def compile_training_components(self, *, dynamic: bool, mode: str) -> None:
        raise NotImplementedError

    def parameter_groups(self) -> tuple[ModelParameterGroup, ...]:
        """Return a complete, disjoint partition of trainable parameters."""

        trainable = tuple(parameter for parameter in self.parameters() if parameter.requires_grad)
        trainable_ids = {id(parameter) for parameter in trainable}
        no_decay_ids = {id(parameter) for parameter in self._no_weight_decay_parameters()}
        if not no_decay_ids <= trainable_ids:
            raise ValueError("no-weight-decay declarations must reference trainable parameters")

        seen: set[int] = set()
        groups: list[ModelParameterGroup] = []
        for name, members in self._parameter_group_members().items():
            parameters = tuple(members)
            if not parameters:
                continue
            ids = [id(parameter) for parameter in parameters]
            if len(ids) != len(set(ids)) or seen.intersection(ids):
                raise ValueError(f"parameter group {name!r} contains duplicate parameters")
            if any(parameter_id not in trainable_ids for parameter_id in ids):
                raise ValueError(f"parameter group {name!r} contains an unknown parameter")
            seen.update(ids)
            decay = tuple(
                parameter
                for parameter in parameters
                if parameter.ndim >= 2 and id(parameter) not in no_decay_ids
            )
            no_decay = tuple(
                parameter
                for parameter in parameters
                if parameter.ndim < 2 or id(parameter) in no_decay_ids
            )
            groups.append(ModelParameterGroup(name=name, decay=decay, no_decay=no_decay))
        if seen != trainable_ids:
            missing = len(trainable_ids - seen)
            raise ValueError(f"parameter groups leave {missing} trainable parameters unassigned")
        return tuple(groups)
