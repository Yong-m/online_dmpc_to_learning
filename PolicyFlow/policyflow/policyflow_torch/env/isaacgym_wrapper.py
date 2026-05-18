from typing import Any, Tuple, Dict
import torch
from policyflow_torch.env import Wrapper


class IsaacGymEnvWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """Isaac Gym environment (preview 3) wrapper

        :param env: The environment to wrap
        :type env: Any supported Isaac Gym environment (preview 3) environment
        """
        super().__init__(env)

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of torch.Tensor and any other info
        """
        observations_dict, reward, terminated, env_info = self._env.step(actions)

        return (
            observations_dict,
            reward,
            terminated,
            env_info,
        )

    def reset(self) -> Tuple[Dict[str, torch.Tensor], Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: torch.Tensor and any other info
        """
        observations_dict, env_info = self._env.reset()
        return observations_dict, env_info

    def render(self, *args, **kwargs) -> None:
        """Render the environment"""
        return None

    def close(self) -> None:
        """Close the environment"""
        pass
