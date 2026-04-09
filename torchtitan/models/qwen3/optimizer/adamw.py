from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from torch.optim import AdamW as TorchAdamW

from torchtitan.distributed import ParallelDims
from torchtitan.experiments.phoenix.optimizer.interfaces import (
    ConfigurableOptimizer,
    ParamGroup,
)

__all__ = [
    "AdamW",
]


class AdamW(TorchAdamW, ConfigurableOptimizer):
    @dataclass(kw_only=True, slots=True)
    class Config(ConfigurableOptimizer.Config):
        lr: float = 8e-4
        # see https://aleph-alpha.atlassian.net/wiki/spaces/RESEARCH/pages/2380267521/Hyperparameter+decisions#weight-decay
        weight_decay: float = 2 ** (-13)
        beta1: float = 0.9
        beta2: float = 0.95
        eps: float = 1e-15
        implementation: Literal["for-loop", "foreach", "fused"] = "fused"

    def __init__(
        self,
        config: Config,
        *,
        param_group: ParamGroup,
        parallel_dims: ParallelDims | None = None,
    ) -> None:
        del parallel_dims

        groups = []
        if param_group.decay_params:
            groups.append(
                {
                    "params": param_group.decay_params,
                    # the adamw implementation multiplies weight decay with lr,
                    # to achieve independent weight decay we need to divide here
                    "weight_decay": config.weight_decay / config.lr,
                }
            )
        if param_group.no_decay_params:
            groups.append({"params": param_group.no_decay_params, "weight_decay": 0.0})

        super().__init__(
            groups,
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            fused=config.implementation == "fused",
            foreach=config.implementation == "foreach",
        )
