import torch, math
from torch import nn
import numpy as np
from typing import Union

def at_least_ndim(
    x: Union[np.ndarray, torch.Tensor, int, float], ndim: int, pad: int = 0
):
    """Add dimensions to the input tensor to make it at least ndim-dimensional.
    Args:
        x: Union[np.ndarray, torch.Tensor, int, float], input tensor
        ndim: int, minimum number of dimensions
        pad: int, padding direction. `0`: pad in the last dimension, `1`: pad in the first dimension

    Returns:
        Any of these 2 options

        - np.ndarray or torch.Tensor: reshaped tensor
        - int or float: input value

    Examples:
        >>> x = np.random.rand(3, 4)
        >>> at_least_ndim(x, 3, 0).shape
        (3, 4, 1)
        >>> x = torch.randn(3, 4)
        >>> at_least_ndim(x, 4, 1).shape
        (1, 1, 3, 4)
        >>> x = 1
        >>> at_least_ndim(x, 3)
        1
    """
    if isinstance(x, np.ndarray):
        if ndim > x.ndim:
            if pad == 0:
                return np.reshape(x, x.shape + (1,) * (ndim - x.ndim))
            else:
                return np.reshape(x, (1,) * (ndim - x.ndim) + x.shape)
        else:
            return x
    elif isinstance(x, torch.Tensor):
        if ndim > x.ndim:
            if pad == 0:
                return torch.reshape(x, x.shape + (1,) * (ndim - x.ndim))
            else:
                return torch.reshape(x, (1,) * (ndim - x.ndim) + x.shape)
        else:
            return x
    elif isinstance(x, (int, float)):
        return x
    else:
        raise ValueError(f"Unsupported type {type(x)}")

# ================= Sampling step schedule ===============
def uniform_sampling_step_schedule(T: int = 1000, sampling_steps: int = 10):
    return torch.linspace(0, T - 1, sampling_steps + 1, dtype=torch.long)


def uniform_sampling_step_schedule_continuous(trange=None, sampling_steps: int = 10):
    if trange is None:
        trange = [1e-3, 1.0]
    return torch.linspace(trange[0], trange[1], sampling_steps + 1, dtype=torch.float32)


def quad_sampling_step_schedule(T: int = 1000, sampling_steps: int = 10, n: int = 1.5):
    schedule = (T - 1) * (
        torch.linspace(0, 1, sampling_steps + 1, dtype=torch.float32) ** n
    )
    return schedule.to(torch.long)


def quad_sampling_step_schedule_continuous(
    trange=None, sampling_steps: int = 10, n: int = 1.5
):
    if trange is None:
        trange = [1e-3, 1.0]
    schedule = (trange[1] - trange[0]) * (
        torch.linspace(0, 1, sampling_steps + 1, dtype=torch.float32) ** n
    ) + trange[0]
    return schedule


def cat_cos_sampling_step_schedule(
    T: int = 1000, sampling_steps: int = 10, n: int = 2.0
):
    idx = torch.linspace(0, 1, sampling_steps + 1, dtype=torch.float32)
    idx = (
        0.5 * (2 * (idx > 0.5) - 1) * torch.sin(np.pi * torch.abs(idx - 0.5)) ** (1 / n)
        + 0.5
    )
    schedule = (T - 1) * idx
    return schedule.to(torch.long)


def cat_cos_sampling_step_schedule_continuous(
    trange=None, sampling_steps: int = 10, n: int = 2.0
):
    if trange is None:
        trange = [1e-3, 1.0]
    idx = torch.linspace(0, 1, sampling_steps + 1, dtype=torch.float32)
    idx = (
        0.5 * (2 * (idx > 0.5) - 1) * torch.sin(np.pi * torch.abs(idx - 0.5)) ** (1 / n)
        + 0.5
    )
    schedule = (trange[1] - trange[0]) * idx + trange[0]
    return schedule


def quad_cos_sampling_step_schedule(
    T: int = 1000, sampling_steps: int = 10, n: int = 2.0
):
    idx = torch.linspace(0, 1, sampling_steps + 1, dtype=torch.float32)
    idx = ((torch.sin(np.pi * (idx - 0.5)) + 1) / 2) ** n
    schedule = (T - 1) * idx
    return schedule.to(torch.long)


def quad_cos_sampling_step_schedule_continuous(
    trange=None, sampling_steps: int = 10, n: int = 2.0
):
    if trange is None:
        trange = [1e-3, 1.0]
    idx = torch.linspace(0, 1, sampling_steps + 1, dtype=torch.float32)
    idx = ((torch.sin(np.pi * (idx - 0.5)) + 1) / 2) ** n
    schedule = (trange[1] - trange[0]) * idx + trange[0]
    return schedule


SAMPLING_STEP_SCHEDULE = {
    "uniform": uniform_sampling_step_schedule,
    "uniform_continuous": uniform_sampling_step_schedule_continuous,
    "quad": quad_sampling_step_schedule,
    "quad_continuous": quad_sampling_step_schedule_continuous,
    "cat_cos": cat_cos_sampling_step_schedule,
    "cat_cos_continuous": cat_cos_sampling_step_schedule_continuous,
    "quad_cos": quad_cos_sampling_step_schedule,
    "quad_cos_continuous": quad_cos_sampling_step_schedule_continuous,
}


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ema_update(model: nn.Module, model_ema: nn.Module, ema_rate: float):
    for param, param_ema in zip(model.parameters(), model_ema.parameters()):
        param_ema.data.mul_(ema_rate).add_(param.data, alpha=1 - ema_rate)


# -----------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures,
# from https://github.com/NVlabs/edm/blob/main/training/networks.py#L269
class PositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_positions: int = 10000, endpoint: bool = False):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.dim // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.dim // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.nn.functional.pad(
            torch.cat([x.cos(), x.sin()], dim=1),
            pad=[0, self.dim - freqs.shape[-1] * 2],
        )
        return x


class UntrainablePositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_positions: int = 10000, endpoint: bool = False):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.dim // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.dim // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = torch.einsum("...i,j->...ij", x, freqs.to(x.dtype))
        # x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


# -----------------------------------------------------------
# Timestep embedding used in Transformer
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.einsum("...i,j->...ij", x, emb.to(x.dtype))
        # emb = x[:, None] * emb[None, :]
        emb = torch.nn.functional.pad(
            torch.cat((emb.sin(), emb.cos()), dim=-1),
            pad=[0, self.dim - emb.shape[-1] * 2],
        )
        return emb


# -----------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures
class FourierEmbedding(nn.Module):
    def __init__(self, dim: int, scale=16):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(dim // 8) * scale, requires_grad=False)
        self.mlp = nn.Sequential(
            nn.Linear(2 * (dim // 8), dim), nn.Mish(), nn.Linear(dim, dim)
        )

    def forward(self, x: torch.Tensor):
        emb = torch.einsum("...i,j->...ij", x, (2 * np.pi * self.freqs).to(x.dtype))
        # emb = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        emb = torch.cat([emb.cos(), emb.sin()], -1)
        return self.mlp(emb)


class UntrainableFourierEmbedding(nn.Module):
    def __init__(self, dim: int, scale=16):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(dim // 2) * scale, requires_grad=False)

    def forward(self, x: torch.Tensor):
        emb = torch.einsum("...i,j->...ij", x, (2 * np.pi * self.freqs).to(x.dtype))
        # emb = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        emb = torch.cat([emb.cos(), emb.sin()], -1)
        return emb


TIMESTEP_EMBEDDING = {
    "positional": PositionalEmbedding,
    "fourier": FourierEmbedding,
    "untrainable_fourier": UntrainableFourierEmbedding,
    "untrainable_positional": UntrainablePositionalEmbedding,
}