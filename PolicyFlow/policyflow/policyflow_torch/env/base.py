from typing import Any, Mapping, Sequence, Tuple, Union, Dict
import gym
import torch


class Wrapper(object):
    def __init__(self, env: Any) -> None:
        """Base wrapper class for RL environments

        :param env: The environment to wrap
        :type env: Any supported RL environment
        """
        self._env = env
        try:
            self._unwrapped = self._env.unwrapped
        except:
            self._unwrapped = env

        # device
        if hasattr(self._unwrapped, "device"):
            self._device = torch.device(self._unwrapped.device)
        else:
            self._device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
    
    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self._env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    def __getattr__(self, key: str) -> Any:
        """Get an attribute from the wrapped environment

        :param key: The attribute name
        :type key: str

        :raises AttributeError: If the attribute does not exist

        :return: The attribute value
        :rtype: Any
        """
        if hasattr(self._env, key):
            return getattr(self._env, key)
        if hasattr(self._unwrapped, key):
            return getattr(self._unwrapped, key)
        raise AttributeError(
            f"Wrapped environment ({self._unwrapped.__class__.__name__}) does not have attribute '{key}'"
        )
    
    @property
    def cfg(self) -> object:
        """Returns the configuration class instance of the environment."""
        return self._unwrapped.cfg
    
    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self._env.render_mode

    def reset(self) -> Tuple[Dict[str, torch.Tensor], Any]:
        """Reset the environment

        :raises NotImplementedError: Not implemented

        :return: Observation, info
        :rtype: torch.Tensor and any other info
        """
        raise NotImplementedError

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :raises NotImplementedError: Not implemented

        :return: Observation_dict, reward, terminated, info
        :rtype: tuple of torch.Tensor and any other info
        """
        raise NotImplementedError

    def render(self, *args, **kwargs) -> Any:
        """Render the environment

        :raises NotImplementedError: Not implemented

        :return: Any value from the wrapped environment
        :rtype: any
        """
        raise NotImplementedError

    def close(self) -> None:
        """Close the environment

        :raises NotImplementedError: Not implemented
        """
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        """The device used by the environment

        If the wrapped environment does not have the ``device`` property, the value of this property
        will be ``"cuda"`` or ``"cpu"`` depending on the device availability
        """
        return self._device

    @property
    def num_envs(self) -> int:
        """Number of environments

        If the wrapped environment does not have the ``num_envs`` property, it will be set to 1
        """
        return self._unwrapped.num_envs if hasattr(self._unwrapped, "num_envs") else 1

    @property
    def observation_space(self) -> gym.Space:
        """Observation space"""
        return self._unwrapped.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Action space"""
        return self._unwrapped.action_space
    
    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__
    
    @property
    def unwrapped(self):
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self._env.unwrapped
    
    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self.unwrapped.episode_length_buf
    
    @property
    def max_episode_length(self):
        """The maximal episode length."""
        return self.unwrapped.max_episode_length

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.

        Note:
            This is needed to perform random initialization of episode lengths in Runner.
        """
        self.unwrapped.episode_length_buf = value
    
    def close(self):  # noqa: D102
        return self._env.close()

