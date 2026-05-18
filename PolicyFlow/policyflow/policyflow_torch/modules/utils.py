from torch import nn
import torch.nn.functional as F
import torch


class AnnealCoefficient(nn.Module):
    def __init__(self, start=0.0, end=1.0, total_steps=10000, mode="linear"):
        """
        Args:
            start (float): initial coefficient
            end (float): final coefficient
            total_steps (int): number of steps to anneal from start to end
            mode (str): annealing mode, choose from ["linear", "exp", "cos"]
        """
        super().__init__()
        self.start = start
        self.end = end
        self.total_steps = total_steps
        self.mode = mode
        # step counter, stored as buffer so it moves with model.to(device)
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))

    def forward(self):
        """Return the current annealing coefficient"""
        t = torch.clamp(self.step.float() / self.total_steps, 0.0, 1.0)

        if self.mode == "linear":
            coeff = self.start + (self.end - self.start) * t
        elif self.mode == "exp":
            # exponential growth curve
            coeff = self.start + (self.end - self.start) * (1 - torch.exp(-5 * t))
        elif self.mode == "cos":
            # cosine schedule
            coeff = self.end - (self.end - self.start) * (
                0.5 * (1 + torch.cos(torch.pi * t))
            )
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        return coeff.detach()

    def step_update(self):
        """Increase step counter by one"""
        self.step += 1


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    elif act_name == "mish":
        return nn.Mish()
    elif act_name == "linear":
        return nn.Identity()
    elif act_name == "softmax":
        return nn.Softmax()
    elif act_name == "silu":
        return nn.SiLU()
    else:
        print("invalid activation function!")
        return None


def init_xavier_uniform(layer, activation):
    try:
        nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain(activation))
    except ValueError:
        nn.init.xavier_uniform_(layer.weight)


def soft_clip(x: torch.Tensor, min_val, max_val):
    return min_val + F.softplus(x - min_val) - F.softplus(x - max_val)
