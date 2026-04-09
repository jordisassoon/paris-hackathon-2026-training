# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

import dataclasses
from dataclasses import dataclass

import torch
import torch.nn as nn

from torchtitan.models.qwen3.optimizer.interfaces import (
    empty_param_groups,
    OptimizerGroup,
    ParamGroup,
)
from torchtitan.models.common.attention import (
    AttentionMasksType,
    GQAttention,
    VarlenAttention,
)
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.tools.logging import logger


class Qwen3TransformerBlock(TransformerBlock):
    """
    Qwen3 TransformerBlock Module

    Args:
        layer_id (int): Identifier for the layer.
        dim (int): Model dimension.
        n_layers (int): Total number of layers.
        config (Qwen3TransformerBlock.Config): Block configuration.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()

        self.attention = config.attention.build()

        self.moe_enabled = config.moe is not None
        if self.moe_enabled:
            assert config.moe is not None
            self.moe = config.moe.build()
        else:
            assert config.feed_forward is not None
            self.feed_forward = config.feed_forward.build()

        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        x = x + self.attention(
            self.attention_norm(x), freqs_cis, attention_masks, positions
        )

        if self.moe_enabled:
            x = x + self.moe(self.ffn_norm(x))
        else:
            x = x + self.feed_forward(self.ffn_norm(x))
        return x

    def get_param_groups(self) -> dict[OptimizerGroup, ParamGroup]:
        if self.moe_enabled:
            raise NotImplementedError(
                "Qwen3TransformerBlock.get_param_groups() supports dense blocks only."
            )

        groups = empty_param_groups()
        for layer_groups in (
            self.attention.get_param_groups(),
            self.feed_forward.get_param_groups(),
        ):
            for key, group in layer_groups.items():
                groups[key].extend(group)

        for norm in (self.attention_norm, self.ffn_norm):
            for param in norm.parameters():
                if param.requires_grad:
                    groups[OptimizerGroup.BACKBONE_1D].no_decay_params.append(param)

        return groups


class Qwen3Model(Decoder):
    """
    Qwen3Model Module

    Args:
        config (Qwen3Model.Config): Model configuration.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 1024
        vocab_size: int = 151936

        def update_from_config(
            self,
            *,
            trainer_config,
            **kwargs,
        ) -> None:

            training = trainer_config.training
            parallelism = trainer_config.parallelism
            debug = trainer_config.debug
            seq_len = training.seq_len
            if seq_len > self.rope.max_seq_len:
                logger.warning(
                    f"Sequence length {seq_len} exceeds original maximum {self.rope.max_seq_len}."
                )
            # Sync rope max_seq_len
            self.rope = dataclasses.replace(self.rope, max_seq_len=seq_len)

            for layer_cfg in self.layers:
                if layer_cfg.moe is not None:
                    layer_cfg.moe.router._debug_force_load_balance = (
                        debug.moe_force_load_balance
                    )

            if parallelism.context_parallel_degree > 1 and isinstance(
                self.layers[0].attention.inner_attention, VarlenAttention.Config
            ):
                raise NotImplementedError(
                    "Context Parallel only supports SDPA and FlexAttention. "
                    "Varlen attention is not supported with CP."
                )

            tp = parallelism.tensor_parallel_degree
            if tp > 1:
                n_heads = self.layers[0].attention.n_heads
                # pyrefly: ignore [missing-attribute]
                n_kv_heads = self.layers[0].attention.n_kv_heads or n_heads
                if n_heads % tp != 0:
                    raise ValueError(
                        f"tensor_parallel_degree ({tp}) must divide n_heads ({n_heads})."
                    )
                if n_kv_heads % tp != 0:
                    raise ValueError(
                        f"tensor_parallel_degree ({tp}) must divide n_kv_heads ({n_kv_heads})."
                    )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:

            assert isinstance(self.layers[0].attention, GQAttention.Config)
            assert self.layers[0].attention.head_dim is not None
            return get_moe_model_nparams_and_flops(
                self,
                model,
                self.layers[0].attention.n_heads,
                2 * self.layers[0].attention.head_dim,
                seq_len,
            )

    def get_param_groups(self) -> dict[OptimizerGroup, ParamGroup]:
        if any(layer_cfg.moe is not None for layer_cfg in self.config.layers):
            raise NotImplementedError(
                "Qwen3Model.get_param_groups() currently supports dense models only."
            )

        groups = empty_param_groups()

        if (
            self.tok_embeddings is not None
            and self.tok_embeddings.weight.requires_grad
        ):
            groups[OptimizerGroup.EMBEDDING].no_decay_params.append(
                self.tok_embeddings.weight
            )

        if self.output is not None:
            if self.output.weight.requires_grad:
                groups[OptimizerGroup.HEADS].decay_params.append(self.output.weight)
            if self.output.bias is not None and self.output.bias.requires_grad:
                groups[OptimizerGroup.HEADS].no_decay_params.append(self.output.bias)

        if self.norm is not None:
            for param in self.norm.parameters():
                if param.requires_grad:
                    groups[OptimizerGroup.BACKBONE_1D].no_decay_params.append(param)

        if self.layers is not None:
            for layer in self.layers.values():
                layer_groups = self._unwrap_grouping_module(layer).get_param_groups()
                for key, group in layer_groups.items():
                    groups[key].extend(group)

        self._validate_param_groups(groups)
        return groups

    def _validate_param_groups(self, groups: dict[OptimizerGroup, ParamGroup]) -> None:
        seen: dict[int, str] = {}
        for group_name, parameter_group in groups.items():
            for param in parameter_group.all_params():
                param_id = id(param)
                if param_id in seen:
                    raise ValueError(
                        f"Parameter appears in both '{seen[param_id]}' and "
                        f"'{group_name.value}' groups"
                    )
                seen[param_id] = group_name.value

        for name, param in self.named_parameters():
            if param.requires_grad and id(param) not in seen:
                raise ValueError(
                    f"Parameter '{name}' (shape {tuple(param.shape)}) requires "
                    f"grad but is not assigned to any optimizer group"
                )

    @staticmethod
    def _unwrap_grouping_module(module: nn.Module) -> nn.Module:
        current = module
        while not hasattr(current, "get_param_groups"):
            children = list(current.children())
            if len(children) != 1:
                raise TypeError(
                    f"Cannot find get_param_groups() under wrapper {type(current).__name__}"
                )
            current = children[0]
        return current
