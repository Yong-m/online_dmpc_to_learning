from typing import Optional, List

import torch
import torch.nn as nn
import numpy as np

from .utils import TIMESTEP_EMBEDDING
from policyflow_torch.modules import Network


class FlowNetBase(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        timestep_emb_type: str = "positional",
        timestep_emb_params: Optional[dict] = None,
    ):
        assert timestep_emb_type in TIMESTEP_EMBEDDING.keys()
        super().__init__()
        timestep_emb_params = timestep_emb_params or {}
        self.map_noise = TIMESTEP_EMBEDDING[timestep_emb_type](
            emb_dim, **timestep_emb_params
        )

    def forward(
        self,
        x: torch.Tensor,
        noise: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ):
        """
        Input:
            x:          (b, horizon, in_dim)
            noise:      (b, )
            condition:  (b, emb_dim) or None / No condition indicates zeros((b, emb_dim))

        Output:
            y:          (b, horizon, in_dim)
        """
        raise NotImplementedError


class FlowMlp(FlowNetBase):
    def __init__(
        self,
        x_dim: int,
        emb_dim: int = 16,
        activations: List[str] = ["relu", "relu", "relu", "tanh"],
        hidden_dims: List[int] = [256, 256, 256],
        timestep_emb_type: str = "positional",
        timestep_emb_params: Optional[dict] = None,
    ):
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)
        self.mlp = Network(x_dim + emb_dim, x_dim, activations, hidden_dims)

    def forward(
        self,
        x: torch.Tensor,
        noise: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ):
        """
        Input:
            x:          (b, x_dim)
            noise:      (b, )
            condition:  (b, emd_dim)

        Output:
            y:          (b, x_dim)
        """
        t = self.map_noise(noise)
        if condition is not None:
            t += condition
        else:
            t += torch.zeros_like(t)
        return self.mlp(torch.cat([x, t], -1))


class ConditionNetBase(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, condition: torch.Tensor):
        raise NotImplementedError


class IdentityCondition(ConditionNetBase):
    def __init__(self):
        super().__init__()

    def forward(self, condition: torch.Tensor):
        return condition
    
class ConditionLinearLayer(ConditionNetBase):
    def __init__(
        self,
        cond_dim: int,
        emb_dim: int = 16,
    ):
        super().__init__()
        self.mlp = torch.nn.Linear(cond_dim, emb_dim)

    def forward(self, condition: torch.Tensor):
        return self.mlp(condition)


class ConditionMlp(ConditionNetBase):
    def __init__(
        self,
        cond_dim: int,
        emb_dim: int = 16,
        activations: List[str] = ["elu", "elu", "elu", "linear"],
        hidden_dims: List[int] = [256, 256, 256],
    ):
        super().__init__()
        self.mlp = Network(cond_dim, emb_dim, activations, hidden_dims)

    def forward(self, condition: torch.Tensor):
        return self.mlp(condition)


class LearnableVariance(nn.Module):
    def __init__(
        self,
        dims: int,
        log_std_max: float = 4.0,
        log_std_min: float = -20.0,
        std_init: float = 1.0,
    ):
        super().__init__()
        self._log_std_max = log_std_max
        self._log_std_min = log_std_min

        self._log_std = nn.Parameter(torch.ones(dims) * np.log(std_init))

    def forward(self):
        return self._log_std.clamp(self._log_std_min, self._log_std_max).exp()

    @property
    def std(self):
        return self._log_std.clamp(self._log_std_min, self._log_std_max).exp()
