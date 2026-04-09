# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any

from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.nn import Module, Parameter
from torch.optim import Optimizer

from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import Configurable
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.phoenix.optimizer.interfaces import (
    ConfigurableOptimizer,
    OptimizerGroup,
    ParamGroup,
)
from torchtitan.tools.logging import logger


__all__ = [
    "MultiGroupOptimizersContainer",
]


class MultiGroupOptimizersContainer(OptimizersContainer):
    """A container for multiple optimizers with different parameter groups.

    This class supports 5 different parameter groups, each with its own optimizer:
    1. embedding: tok_embeddings parameters
    2. heads: output/lm_head and router parameters
    3. backbone_2d: 2D weight matrices in the backbone (attention, FFN projections)
    4. backbone_1d: 1D parameters in the backbone (biases, norms, etc.)

     Within each group, parameters are further split by weight-decay intent:
    the model's Phoenix.get_param_groups returns
    ParamGroup instances that distinguish decay vs no-decay params.

    This allows using different optimizers for different groups, e.g. Muon for
    backbone 2-D params and AdamW for the rest.

    Args:
        model_parts: List of model parts to be optimized.
        parallel_dims: Parallel dimensions for distributed training.
        config: Typed optimizer configuration with per-group settings.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        # This replaces the base OptimizersContainer.Config entirely because
        # Phoenix configures optimizers per parameter group rather than via one
        # shared optimizer hyperparameter set.
        embedding: ConfigurableOptimizer.Config
        backbone_1d: ConfigurableOptimizer.Config
        backbone_2d: ConfigurableOptimizer.Config
        heads: ConfigurableOptimizer.Config

    def __init__(
        self,
        config: Config,
        *,
        model_parts: list[Module],
        parallel_dims: ParallelDims,
    ) -> None:
        all_params: list = []
        self.optimizers: list[Optimizer] = []
        self.model_parts = model_parts

        self.group_optimizers: dict[OptimizerGroup, list[tuple[int, Optimizer]]] = {
            group: [] for group in OptimizerGroup
        }

        # Import Phoenix here to avoid circular import
        from torchtitan.experiments.phoenix.feed_forward.router import (
            TokenChoiceTopKRouter,
        )
        from torchtitan.experiments.phoenix.model.model import Phoenix

        for part_idx, model in enumerate(self.model_parts):
            param_groups: dict[OptimizerGroup, ParamGroup] = model.get_param_groups()

            for group in OptimizerGroup:
                pg = param_groups[group]
                if pg:
                    group_config = getattr(config, group.value)
                    logger.info(f"Optimizer config for {group.value}: {group_config}")
                    opt = group_config.build(
                        param_group=pg,
                        parallel_dims=parallel_dims,
                    )
                    self.group_optimizers[group].append((part_idx, opt))
                    self.optimizers.append(opt)
                    all_params.extend(pg.all_params())

        self._log_param_counts()
        self._validate_length(len(self.optimizers))
        # We need to call the base class __init__ to initialize the optimizer hooks.
        self._post_init(all_params, {})

    def _log_param_counts(self) -> None:
        total_params = 0
        for group in OptimizerGroup:
            group_params = 0
            for _, opt in self.group_optimizers[group]:
                group_params += sum(
                    p.numel() for pg in opt.param_groups for p in pg["params"]
                )
            total_params += group_params
            if group_params > 0:
                logger.info(
                    f"MultiGroupOptimizer: {group.value} has {group_params:,} params"
                )
        logger.info(f"MultiGroupOptimizer: total {total_params:,} params")

    def state_dict(self) -> dict[str, Any]:
        state = {}
        for group in OptimizerGroup:
            for part_idx, opt in self.group_optimizers[group]:
                model = self.model_parts[part_idx]
                group_sd = get_optimizer_state_dict(
                    model,
                    opt,
                    options=StateDictOptions(flatten_optimizer_state_dict=True),
                )
                state.update(group_sd)
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for group in OptimizerGroup:
            for part_idx, opt in self.group_optimizers[group]:
                model = self.model_parts[part_idx]
                set_optimizer_state_dict(
                    model,
                    opt,
                    optim_state_dict=state_dict,
                    options=StateDictOptions(flatten_optimizer_state_dict=True),
                )
