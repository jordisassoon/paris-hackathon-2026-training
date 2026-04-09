from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from torch.nn import Parameter
from torch.optim import Optimizer

from torchtitan.config import Configurable
from torchtitan.distributed import ParallelDims

__all__ = [
    "ConfigurableOptimizer",
    "OptimizerGroup",
    "ParamGroup",
]


class OptimizerGroup(StrEnum):
    EMBEDDING = "embedding"
    BACKBONE_1D = "backbone_1d"
    BACKBONE_2D = "backbone_2d"
    HEADS = "heads"


@dataclass
class ParamGroup:
    """Parameters for an optimizer group, split by weight decay behavior.

    Attributes:
        decay_params: Parameters that receive weight decay.
        no_decay_params: Parameters that skip weight decay.
    """

    decay_params: list[Parameter] = field(default_factory=list)
    no_decay_params: list[Parameter] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.decay_params or self.no_decay_params)

    def all_params(self) -> list[Parameter]:
        return self.decay_params + self.no_decay_params

    def extend(self, other: ParamGroup) -> None:
        self.decay_params.extend(other.decay_params)
        self.no_decay_params.extend(other.no_decay_params)


class ConfigurableOptimizer(Configurable):
    """Base class for Qwen3 optimizers built from a nested Config."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        def build(
            self,
            *,
            param_group: ParamGroup,
            parallel_dims: ParallelDims | None = None,
            **kwargs,
        ) -> Optimizer:
            if self._owner is None:
                raise NotImplementedError(
                    f"{type(self).__name__} has no owner class. "
                    "Define Config inside a Configurable subclass."
                )
            return self._owner(
                config=self,
                param_group=param_group,
                parallel_dims=parallel_dims,
            )
