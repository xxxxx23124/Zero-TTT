"""Concrete policy-value Transformer assembly."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
from torch import nn

from zero_ttt.config import ExecutionConfig, ModelConfig
from zero_ttt.game.rules import ACTION_SIZE, BOARD_AREA, BOARD_SIZE
from zero_ttt.model.base import BasePolicyValueModel
from zero_ttt.model.blocks import TransformerBackbone
from zero_ttt.model.components import GoTokenEncoder, PolicyValueHeads
from zero_ttt.model.contracts import BackboneOutput, ModelPredictions
from zero_ttt.model.depth_mixing import IdentityDepthMixer, SparseDepthWeightedAverage
from zero_ttt.model.execution import BlockExecutor
from zero_ttt.model.hypernet import SharedDynamicLowRank
from zero_ttt.model.interfaces import NoOpBlockResidualPlugin
from zero_ttt.model.tokens import TokenLayout


def eager_execution_config() -> ExecutionConfig:
    return ExecutionConfig(
        activation_checkpoint=False,
        activation_checkpoint_stride=1,
        compile_model=False,
        compile_mode="default",
    )


class PolicyValueTransformer(BasePolicyValueModel):
    def __init__(
        self,
        config: ModelConfig,
        execution: ExecutionConfig | None = None,
    ) -> None:
        super().__init__(
            input_planes=config.input_planes,
            global_features=config.global_features,
            board_size=BOARD_SIZE,
            action_size=ACTION_SIZE,
        )
        self.config = config
        self.layout = TokenLayout(board_tokens=BOARD_AREA, special_tokens=1)
        self.encoder = GoTokenEncoder(config, self.layout)
        hyper = config.hypernet
        block_plugin = (
            SharedDynamicLowRank(
                d_model=config.d_model,
                d_ff=config.d_ff,
                rank=hyper.rank,
                hidden_dim=hyper.hidden_dim,
                n_layers=config.n_layers,
                context_gradient_scale=hyper.context_gradient_scale,
            )
            if hyper.enabled
            else NoOpBlockResidualPlugin()
        )
        depth_mixer = (
            SparseDepthWeightedAverage(config.n_layers, config.depth_mixing)
            if config.depth_mixing.enabled
            else IdentityDepthMixer()
        )
        self.backbone = TransformerBackbone(
            config,
            execution or eager_execution_config(),
            self.layout,
            block_plugin,
            depth_mixer,
        )
        self.heads = PolicyValueHeads(config, self.layout)

    def _encode_tokens(
        self,
        board_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(board_features, global_features)

    def _forward_backbone(self, hidden: torch.Tensor) -> BackboneOutput:
        return self.backbone(hidden)

    def _decode_outputs(self, hidden: torch.Tensor) -> ModelPredictions:
        return self.heads(hidden)

    def _parameter_group_members(self) -> Mapping[str, Iterable[nn.Parameter]]:
        plugin_parameters = tuple(self.backbone.block_plugin.parameters())
        plugin_ids = {id(parameter) for parameter in plugin_parameters}
        base_parameters = tuple(
            parameter for parameter in self.parameters() if id(parameter) not in plugin_ids
        )
        groups: dict[str, tuple[nn.Parameter, ...]] = {"base": base_parameters}
        if plugin_parameters:
            group_name = self.backbone.block_plugin.optimization_group
            if group_name is None:
                raise ValueError("trainable block plugins must declare an optimization group")
            groups[group_name] = plugin_parameters
        return groups

    def _no_weight_decay_parameters(self) -> Iterable[nn.Parameter]:
        return self.encoder.no_weight_decay_parameters()

    def configure_execution(self, config: ExecutionConfig) -> None:
        self.backbone.executor = BlockExecutor(config)

    def compile_training_components(self, *, dynamic: bool, mode: str) -> None:
        self.backbone.compile_blocks(dynamic=dynamic, mode=mode)

    @property
    def block_plugin(self):
        return self.backbone.block_plugin

    @property
    def depth_mixer(self):
        return self.backbone.depth_mixer
