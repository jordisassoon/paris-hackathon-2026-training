from __future__ import annotations

from dataclasses import dataclass

from dion import Muon as DionMuon

from torchtitan.distributed import ParallelDims
from torchtitan.experiments.phoenix.optimizer.interfaces import (
    ConfigurableOptimizer,
    ParamGroup,
)

__all__ = [
    "Muon",
]


class Muon(DionMuon, ConfigurableOptimizer):
    @dataclass(kw_only=True, slots=True)
    class Config(ConfigurableOptimizer.Config):
        lr: float = 8e-2
        # see https://aleph-alpha.atlassian.net/wiki/spaces/RESEARCH/pages/2380267521/Hyperparameter+decisions#weight-decay
        weight_decay: float = 2 ** (-13)
        nesterov: bool = True

    def __init__(
        self,
        config: Config,
        *,
        param_group: ParamGroup,
        parallel_dims: ParallelDims | None = None,
    ) -> None:
        if parallel_dims is None:
            raise TypeError("Muon requires parallel_dims")
        if param_group.no_decay_params:
            raise ValueError("MuonOptimizer does not support no-decay parameters")

        super().__init__(
            param_group.decay_params,
            lr=config.lr,
            # the muon implementation multiplies weight decay with lr,
            # to achieve independent weight decay we need to divide here
            weight_decay=config.weight_decay / config.lr,
            nesterov=config.nesterov,
            flatten=False,
            distributed_mesh=parallel_dims.get_optional_mesh("fsdp"),
        )
