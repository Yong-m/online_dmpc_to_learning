from __future__ import annotations
from abc import ABC, abstractmethod
import torch
from typing import Any, Mapping, Optional, Union, Dict, Tuple, Callable
from policyflow_torch.storage import ReplayBuffer
from policyflow_torch.utils.benchmarkable import Benchmarkable
from policyflow_torch.utils.serializable import Serializable


class Agent(ABC, Benchmarkable, Serializable):
    def __init__(
        self,
        models: Mapping[str, Any],
        replay_buffer: ReplayBuffer,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
        benchmark: bool = False,
    ) -> None:

        super().__init__()

        self.model_dict = models
        self.replay_buffer = replay_buffer
        self.device = (
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if device is None
            else torch.device(device)
        )
        self.cfg = cfg if cfg is not None else {}

        self._register_serializable("model_dict")

        self._bm_toggle(benchmark)

    def set_mode(self, mode: str) -> None:
        """Set the model mode (training or evaluation)

        :param mode: Mode: 'train' for training or 'eval' for evaluation
        :type mode: str
        """
        for model in self.model_dict.values():
            if model is not None:
                if mode == "train":
                    model.train()
                elif mode == "eval":
                    model.eval()

    @abstractmethod
    def draw_actions(
        self, obs_dict: Dict[str, torch.Tensor], env_info: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Union[Dict[str, torch.Tensor], None]]:
        """Draws actions from the action space.

        Args:
            obs (torch.Tensor): The observations for which to draw actions.
            env_info (Dict[str, Any]): The environment information for the observations.
        Returns:
            A tuple containing the actions and the data dictionary.
        """
        pass

    @abstractmethod
    def process_transition(
        self,
        observations_dict: Dict[str, torch.Tensor],
        environement_info: Dict[str, Any],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations_dict: Dict[str, torch.Tensor],
        dones: torch.Tensor,
        actions_info: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Processes a transition before it is added to the replay memory.

        Args:
            observations_dict (Dict[str, torch.Tensor]): The observations from the environment.
            environment_info (Dict[str, Any]): The environment information.
            actions (torch.Tensor): The actions computed by the actor.
            rewards (torch.Tensor): The rewards from the environment.
            next_observations_dict (Dict[str, torch.Tensor]): The next observations from the environment.
            next_environment_info (Dict[str, Any]): The next environment information.
            dones (torch.Tensor): The done flags from the environment.
            data (Dict[str, torch.Tensor]): Additional data to include in the transition.
        Returns:
            A dictionary containing the processed transition.
        """
        pass

    @abstractmethod
    def update(self) -> Dict[str, Union[float, torch.Tensor]]:
        """Updates the agent's parameters.

        Args:
            dataset (Dataset): The dataset from which to update the agent.
        Returns:
            A dictionary containing the loss values.
        """
        pass

    @abstractmethod
    def to(self, device: str) -> Agent:
        """Transfers agent parameters to device."""
        self.device = device
        return self

    @abstractmethod
    def export_onnx(self) -> Tuple[torch.nn.Module, torch.Tensor, Dict]:
        """Exports the agent's policy network to ONNX format.

        Returns:
            A tuple containing the ONNX model, the input arguments, and the keyword arguments.
        """
        pass

    @abstractmethod
    def get_inference_policy(self, device: str = None) -> Callable:
        """Returns a function that computes actions from observations without storing gradients.

        Args:
            device (torch.device): The device to use for inference.
        Returns:
            A function that computes actions from observations.
        """
        pass
