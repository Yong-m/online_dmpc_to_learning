from typing import Optional, Union, Callable

import torch
import torch.nn as nn
import copy
import math

from .flow_net import (
    ConditionNetBase,
    IdentityCondition,
    FlowNetBase,
    LearnableVariance,
)
from .utils import at_least_ndim, SAMPLING_STEP_SCHEDULE


class ContinuousNormalizingFlow:
    def __init__(
        self,
        x_dims,
        # ----------------- Neural Networks ----------------- #
        nn_flow: FlowNetBase,
        nn_condition: Optional[ConditionNetBase] = None,
        # ------------------ Training Params ---------------- #
        ema_rate: float = 0.995,
        using_ema: bool = False,
        sample_steps: int = 10,
        sample_step_schedule: Union[str, Callable] = "uniform_continuous",
        interpolation_type: str = "rectified_flow",  # stochastic_interpolant, trigflow, rectified_flow
        device: Union[torch.device, str] = "cpu",
    ):
        self.device = device
        self.ema_rate = ema_rate
        self.using_ema = using_ema
        self.sample_steps = sample_steps
        self.interpolation_type = interpolation_type

        # ===================== Sampling Schedule ====================
        if (
            interpolation_type == "stochastic_interpolant"
            or interpolation_type == "rectified_flow"
        ):
            final_t = 1.0
        elif interpolation_type == "trigflow":
            final_t = math.pi / 2.0
        else:
            raise ValueError(
                f"Interpolation type {interpolation_type} is not supported."
            )
        if isinstance(sample_step_schedule, str):
            if sample_step_schedule in SAMPLING_STEP_SCHEDULE.keys():
                self.sample_step_schedule = SAMPLING_STEP_SCHEDULE[
                    sample_step_schedule
                ]([0.0, final_t], self.sample_steps)
            else:
                raise ValueError(
                    f"Sampling step schedule {sample_step_schedule} is not supported."
                )
        elif callable(sample_step_schedule):
            self.sample_step_schedule = sample_step_schedule(
                [0.0, final_t], self.sample_steps
            )
        else:
            raise ValueError("sample_step_schedule must be a callable or a string")

        time_steps = []
        for i in range(self.sample_steps):
            t = self.sample_step_schedule[i]
            time_steps.append(t)
            delta_t = self.sample_step_schedule[i + 1] - self.sample_step_schedule[i]
            time_steps.append(t + delta_t / 2)
        time_steps.append(self.sample_step_schedule[self.sample_steps])
        self.time_steps_tensor = torch.tensor(
            time_steps, dtype=torch.float32, device=self.device
        )
        print(f"time_steps_tensor: {self.time_steps_tensor}")

        # nn_condition is None means that the model is not conditioned on any input.
        if nn_condition is None:
            nn_condition = IdentityCondition()

        self.model = nn.ModuleDict(
            {
                "flow": nn_flow.to(self.device),
                "condition": nn_condition.to(self.device),
                "variance": LearnableVariance(dims=x_dims).to(self.device),
            },
        )
        self.model_ema = copy.deepcopy(self.model).requires_grad_(False)
        self.model_last = copy.deepcopy(self.model).requires_grad_(False)

        self.model.train()
        self.model_ema.eval()
        self.model_last.eval()

        # EWMA proximal model (initialized via init_proximal() when needed)
        self.model_proximal = None
        self._ewma_total_weight = 1.0
        self.beta_prox = 0.889

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def ema_update(self):
        with torch.no_grad():
            for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
                p_ema.data.mul_(self.ema_rate).add_(p.data, alpha=1.0 - self.ema_rate)

    def init_proximal(self, beta_prox: float = 0.889):
        """Initialize EWMA proximal model for decoupled policy objectives (PPO-EWMA).

        The proximal model is an exponentially-weighted moving average of the current
        model, updated after each gradient step. It serves as the trust-region anchor
        in the decoupled clipped objective, separate from the behavior policy (model_last).

        Reference: Hilton et al., "Batch size-invariance for policy optimization", NeurIPS 2022.

        Args:
            beta_prox: EWMA decay rate. Higher = slower tracking. Default 0.889 from paper.
        """
        self.beta_prox = beta_prox
        self.model_proximal = copy.deepcopy(self.model).requires_grad_(False)
        self.model_proximal.eval()
        self._ewma_total_weight = 1.0

    def update_proximal(self):
        """Update proximal model via unbiased EWMA. Call after each gradient step.

        Uses bias-corrected EWMA from the ppo-ewma reference implementation:
            w_new = β * w + 1
            θ_prox = (β * w / w_new) * θ_prox + (1 / w_new) * θ
        """
        if self.model_proximal is None:
            return
        beta = self.beta_prox
        new_w = beta * self._ewma_total_weight + 1.0
        decay_ratio = beta * self._ewma_total_weight / new_w
        new_ratio = 1.0 / new_w
        for p, p_prox in zip(self.model.parameters(), self.model_proximal.parameters()):
            p_prox.data.mul_(decay_ratio).add_(p.data, alpha=new_ratio)
        for b, b_prox in zip(self.model.buffers(), self.model_proximal.buffers()):
            b_prox.data.copy_(b.data)
        self._ewma_total_weight = new_w

    def reset_proximal(self):
        """Reset proximal model to current model weights (e.g., at phase boundaries)."""
        if self.model_proximal is None:
            return
        self.model_proximal.load_state_dict(self.model.state_dict())
        self._ewma_total_weight = 1.0

    def save(self, path: str):
        torch.save(
            {
                "model": self.model.state_dict(),
                "model_ema": self.model_ema.state_dict(),
                "model_last": self.model_last.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model_ema.load_state_dict(checkpoint["model_ema"])
        self.model_last.load_state_dict(checkpoint["model_last"])

    def compute_flow_variation(
        self, x1, condition, x0=None, compute_brownian_reg_loss=False,
        return_proximal_info=False,
    ):
        # x0 is the samples of source distribution.
        # x0 x1 is None, then we assume x1 is from a standard Gaussian distribution.
        if x0 is None:
            x0 = torch.randn_like(x1)
        else:
            assert x0.shape == x1.shape, "x0 and x1 must have the same shape"

        # t = torch.rand((x1.shape[0],), device=self.device)

        idx = torch.randint(
            low=0,
            high=self.time_steps_tensor.shape[0],
            size=(x1.shape[0],),
            device=self.device,
        )
        t = self.time_steps_tensor[idx]  # shape: (batch_size,)

        alpha = at_least_ndim(t, x1.dim())

        if self.interpolation_type == "rectified_flow":
            xt = (1.0 - alpha) * x0 + alpha * x1
        elif self.interpolation_type == "stochastic_interpolant":
            xt = (
                (1.0 - alpha) * x0
                + alpha * x1
                + torch.sqrt(2.0 * alpha * (1.0 - alpha).clip(min=1e-6))
                * torch.randn_like(x1)
            )
        elif self.interpolation_type == "trigflow":
            xt = torch.cos(alpha) * x0 + torch.sin(alpha) * x1
        else:
            raise ValueError(
                f"Interpolation type {self.interpolation_type} is not supported."
            )

        # [NaN_DEBUG] — one-shot diagnostics on the first NaN seen anywhere.
        def _nf_stats(name, t_):
            if not torch.isfinite(t_).all():
                nf = (~torch.isfinite(t_)).sum().item()
                finite = t_[torch.isfinite(t_)]
                rng = (finite.min().item(), finite.max().item()) if finite.numel() else ("-", "-")
                print(f"[NaN_DEBUG] {name}: shape={tuple(t_.shape)} non_finite={nf} finite_range={rng}")
                return True
            return False

        if (_nf_stats("x0", x0) or _nf_stats("x1", x1)
                or _nf_stats("condition_input", condition) or _nf_stats("xt", xt)):
            raise RuntimeError("[NaN_DEBUG] non-finite input to flow forward — see above.")

        with torch.inference_mode():
            condition_embeded_last = self.model_last["condition"](condition)
            if _nf_stats("condition_embeded_last", condition_embeded_last):
                raise RuntimeError("[NaN_DEBUG] model_last.condition produced non-finite output.")
            vel_field_last = self.model_last["flow"](
                xt, t, condition_embeded_last
            ).detach()
            if _nf_stats("vel_field_last", vel_field_last):
                raise RuntimeError("[NaN_DEBUG] model_last.flow produced non-finite output.")

        condition_embeded = self.model["condition"](condition)
        if _nf_stats("condition_embeded", condition_embeded):
            raise RuntimeError("[NaN_DEBUG] model.condition produced non-finite output.")
        vel_field = self.model["flow"](xt, t, condition_embeded)
        if _nf_stats("vel_field", vel_field):
            raise RuntimeError("[NaN_DEBUG] model.flow produced non-finite output.")

        delta_vel = vel_field - vel_field_last
        std = torch.ones_like(x1) * self.model["variance"].std

        # Compute proximal policy info for decoupled objective (PPO-EWMA)
        proximal_info = None
        if return_proximal_info and self.model_proximal is not None:
            with torch.no_grad():
                cond_prox = self.model_proximal["condition"](condition)
                vel_prox = self.model_proximal["flow"](xt, t, cond_prox)
            # proximal delta_vel relative to behavior, and proximal std
            proximal_info = (
                (vel_prox - vel_field_last).detach(),
                self.model_proximal["variance"].std.detach(),
            )

        if compute_brownian_reg_loss:
            beta = 1.0 # temperature coeff
            if self.interpolation_type == "rectified_flow":
                brownian_reg_loss = torch.nn.functional.mse_loss(
                    (1 - alpha) * vel_field, beta * (xt - alpha * vel_field_last)
                )
            elif self.interpolation_type == "stochastic_interpolant":
                brownian_reg_loss = torch.nn.functional.mse_loss(
                    (2.0 * ((alpha - 0.5) ** 2) + 0.5) * vel_field,
                    beta * (xt - alpha * vel_field_last),
                )
            elif self.interpolation_type == "trigflow":
                brownian_reg_loss = torch.nn.functional.mse_loss(
                    torch.cos(alpha) * vel_field,
                    beta * (torch.cos(alpha) * xt - torch.sin(alpha) * vel_field_last),
                )
            if return_proximal_info:
                return delta_vel, std, brownian_reg_loss, proximal_info
            return delta_vel, std, brownian_reg_loss
        else:
            if return_proximal_info:
                return delta_vel, std, proximal_info
            return delta_vel, std

    def update(self):
        # model_last buffers (e.g. emp_norm running stats) may become inference tensors
        # after being used inside torch.inference_mode(). load_state_dict does in-place
        # copy, which requires being inside inference_mode to update inference tensors.
        with torch.inference_mode():
            if self.using_ema:
                self.model_last.load_state_dict(self.model_ema.state_dict())
            else:
                self.model_last.load_state_dict(self.model.state_dict())

    # ==================== Sampling: Solving a straight ODE flow ======================
    def sample(
        self,
        x0: torch.Tensor,
        condition: torch.Tensor,
        # ----------------- sampling ----------------- #
        n_samples: int = 1,
    ):
        x0 = x0.to(self.device)

        model = self.model if not self.using_ema else self.model_ema

        xt = x0.clone()
        condition_embeded = model["condition"](condition)

        for i in range(self.sample_steps):
            t = torch.full(
                (n_samples,),
                self.sample_step_schedule[i],
                dtype=torch.float32,
                device=self.device,
            )

            delta_t = self.sample_step_schedule[i + 1] - self.sample_step_schedule[i]
            vel_t = model["flow"](xt, t, condition_embeded)
            xt_middle = xt + vel_t * delta_t / 2
            vel_t = model["flow"](xt_middle, t + delta_t / 2, condition_embeded)
            xt = xt + delta_t * vel_t

        std = torch.ones_like(xt) * model["variance"].std
        return xt.detach(), std.detach()
