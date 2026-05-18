import torch
from torch import nn

class StereographicSphereNormalizer(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        """
        Maps x ∈ ℝ^n to unit sphere S^n ⊂ ℝ^{n+1} via stereographic projection
        Input:
            x: Tensor [B, n]
        Output:
            x_hat: Tensor [B, n+1], unit norm
        """
        norm_sq = torch.sum(x ** 2, dim=-1, keepdim=True)  # [B, 1]
        denom = norm_sq + 1.0 + self.eps                   # avoid divide-by-zero
        x_proj = 2 * x / denom                             # [B, n]
        x_last = (norm_sq - 1.0) / denom                   # [B, 1]
        return torch.cat([x_proj, x_last], dim=-1)         # [B, n+1]

    def inverse(self, x_hat):
        """
        Inverts stereographic projection from S^n ⊂ ℝ^{n+1} back to ℝ^n
        Input:
            x_hat: Tensor [B, n+1]
        Output:
            x: Tensor [B, n]
        """
        x_proj = x_hat[:, :-1]                             # [B, n]
        x_last = x_hat[:, -1:]                             # [B, 1]
        denom = 1.0 - x_last
        denom = torch.clamp(denom, min=self.eps)           # avoid divide-by-zero
        return x_proj / denom                              # [B, n]

class EmpiricalMinMaxNormalizer(nn.Module):
    """
    normalizes data through maximum and minimum expansion.
    """

    def __init__(
        self,
        shape,
        device,
        eps=1e-6,
    ):
        super().__init__()

        self.register_buffer("_min", torch.zeros(shape, device=device).unsqueeze(0))
        self.register_buffer("_max", torch.zeros(shape, device=device).unsqueeze(0))

        self.eps = eps
        self.device = device
        self.count = 0
        self.initialized = False

    @property
    def min(self) -> torch.Tensor:
        return self._min.squeeze(0).detach().clone()

    @property
    def max(self) -> torch.Tensor:
        return self._max.squeeze(0).detach().clone()

    def forward(self, x, update=False) -> torch.Tensor:
        x = x.to(device=self.device)

        if not self.initialized:
            self._update(x)
            self.initialized = True

        if update:
            self._update(x)

        x_normalized = (x - self._min) / (torch.abs(self._max - self._min) + self.eps)
        # normalize to [-1, 1]
        x_normalized = x_normalized * 2.0 - 1.0
        return x_normalized.detach()

    @torch.jit.unused
    def _update(self, x: torch.Tensor) -> None:
        x = x.detach()

        count_x = x.shape[0]
        self.count += count_x
        rate = count_x / self.count

        if count_x > 2:
            q_range = torch.tensor([0.05, 0.95]).to(self.device)
            minmax = torch.quantile(input=x, q=q_range, dim=0)
            delta_min = minmax[0] - self._min
            delta_max = minmax[1] - self._max
            self._min += rate * delta_min
            self._max += rate * delta_max
        else:
            delta_min = torch.min(x, dim=0).values - self._min
            delta_max = torch.max(x, dim=0).values - self._max
            self._min += rate * delta_min
            self._max += rate * delta_max

    @torch.jit.unused
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        y = (y + 1.0) / 2.0
        inv = y * (torch.abs(self._max - self._min) + self.eps) + self._min
        return inv


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape, eps=1e-2, until=None):
        """Initialize EmpiricalNormalization module.

        Args:
            shape (int or tuple of int): Shape of input values except batch axis.
            eps (float): Small value for stability.
            until (int or None): If this arg is specified, the module learns input values until the sum of batch sizes
            exceeds it.

        Note: The normalization parameters are computed over the whole batch, not for each environment separately.
        """
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    def forward(self, x):
        """Normalize mean and variance of values based on empirical values."""

        return (x - self._mean) / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x):
        """Learn input values without computing the output values of them"""
        if self.until is not None and self.count >= self.until:
            return

        count_x = x.shape[0]
        self.count += count_x
        rate = count_x / self.count
        var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
        mean_x = torch.mean(x, dim=0, keepdim=True)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
        self._std = torch.sqrt(self._var)

    @torch.jit.unused
    def inverse(self, y):
        """De-normalize values based on empirical values."""

        return y * (self._std + self.eps) + self._mean