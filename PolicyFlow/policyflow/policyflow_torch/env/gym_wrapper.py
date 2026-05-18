from typing import Any, Tuple, Dict
import torch
from .base import Wrapper
import gymnasium as gym
import gymnasium_robotics
import numpy as np

gym.register_envs(gymnasium_robotics)


class GymEnvWrapper(Wrapper):
    def __init__(self, id_str, num_envs=1) -> None:
        """Isaac Gym environment (preview 3) wrapper

        :param env: The environment to wrap
        :type env: Any supported Isaac Gym environment (preview 3) environment
        """
        env = gym.make_vec(id_str, num_envs, vectorization_mode="sync")
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
        actions_np = actions.cpu().numpy()
        obs, reward, terminated, truncated, env_info = self._env.step(actions_np)

        # next two lines only for maze env
        reward = 0.1 * reward
        obs = np.concatenate((obs["observation"], obs["desired_goal"]), axis=-1)

        # reward = self._env.envs[0].unwrapped.dt * reward

        info = {}
        info["log"] = {}
        for key, value in env_info.items():
            info["log"][key] = torch.from_numpy(value).float().to(device=self.device)

        obs_tensor = torch.from_numpy(obs).float().to(device=self.device)
        reward_tensor = torch.from_numpy(reward).float().to(device=self.device)
        terminated_tensor = torch.from_numpy(terminated).to(device=self.device)

        info["time_outs"] = torch.from_numpy(truncated).to(device=self.device)

        observations_dict = {
            "actor_observations": obs_tensor.clone(),
            "critic_observations": obs_tensor.clone(),
        }

        return (
            observations_dict,
            reward_tensor,
            terminated_tensor,
            info,
        )

    def reset(self) -> Tuple[Dict[str, torch.Tensor], Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: torch.Tensor and any other info
        """
        obs, env_info = self._env.reset()
        obs = np.concatenate(
            (obs["observation"], obs["desired_goal"]), axis=-1
        )  # only for maze env
        info = {}
        info["log"] = {}
        for key, value in env_info.items():
            info["log"][key] = torch.from_numpy(value).float().to(device=self.device)
        obs_tensor = torch.from_numpy(obs).float().to(device=self.device)
        observations_dict = {
            "actor_observations": obs_tensor.clone(),
            "critic_observations": obs_tensor.clone(),
        }
        return observations_dict, info

    def render(self, *args, **kwargs) -> None:
        """Render the environment"""
        return None

    def close(self) -> None:
        """Close the environment"""
        pass
