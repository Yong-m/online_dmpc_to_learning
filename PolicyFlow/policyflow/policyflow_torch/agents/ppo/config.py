from dataclasses import MISSING
from isaaclab.utils import configclass

@configclass
class PPOCfg:
    """Configuration for the PPO algorithm."""

    desired_kl: float = MISSING
    """The desired KL divergence."""    
    
    learning_rate: float = MISSING
    """The initial learning rate."""

    discount_factor: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    time_limit_bootstrap: bool = MISSING
    """Time limit bootstrap for Generalized Advantage Estimation (GAE)."""

    mini_batches: int = MISSING
    """The number of mini-batches per update."""

    learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    entropy_loss_scale: float = MISSING
    """The coefficient for the entropy loss."""

    ratio_clip: float = MISSING
    """The clipping parameter for the policy."""

    clip_predicted_values: bool = MISSING
    """The clipping parameter for the critic."""

    value_clip: float = MISSING
    """The clipping parameter for the critic."""

    value_loss_scale: float = MISSING
    """The coefficient for the value loss."""

    grad_norm_clip: float = MISSING
    """The maximum gradient norm."""


@configclass
class PPOCfgInstance(PPOCfg):
    desired_kl = 0.01
    learning_rate = 2e-4
    discount_factor = 0.99
    lam = 0.95
    time_limit_bootstrap = True
    mini_batches = 4
    learning_epochs = 5
    entropy_loss_scale = 0.01
    ratio_clip = 0.2
    clip_predicted_values = True
    value_clip = 0.2
    value_loss_scale = 1.0
    grad_norm_clip = 1.0