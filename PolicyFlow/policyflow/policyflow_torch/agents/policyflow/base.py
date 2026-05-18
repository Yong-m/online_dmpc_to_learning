import torch
from policyflow_torch.modules import Network, ContinuousNormalizingFlow
from typing import Any, Mapping, Optional, Union
from policyflow_torch.storage import ReplayBuffer
from policyflow_torch.agents import Agent


class PolicyFlowBase(Agent):
    def __init__(
        self,
        models: Mapping[str, Union[ContinuousNormalizingFlow | Network | torch.nn.Module]],
        replay_buffer: ReplayBuffer,
        cfg: dict = dict(),
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Base class that represent a RL agent

        :param models: Models used by the agent
        :param replay_buffer: ReplayBuffer to storage the transitions.
                       If it is a tuple, the first element will be used for training and
                       for the rest only the environment transitions will be added
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :param cfg: Configuration dictionary
        """
        assert "actor" in models.keys(), "no actor policy network in models"
        assert "critic" in models.keys(), "no critic network in models"

        super().__init__(models, replay_buffer, device, cfg)

        # convert the models to their respective device
        for model in self.model_dict.values():
            if model is not None and hasattr(model, "to"):
                model.to(self.device)

    def __str__(self) -> str:
        """Generate a representation of the agent as string

        :return: Representation of the agent as string
        :rtype: str
        """
        string = f"Agent: {repr(self)}"
        for k, v in self.cfg.items():
            if type(v) is dict:
                string += f"\n  |-- {k}"
                for k1, v1 in v.items():
                    string += f"\n  |     |-- {k1}: {v1}"
            else:
                string += f"\n  |-- {k}: {v}"
        return string

    def _empty_preprocessor(self, _input: Any, *args, **kwargs) -> Any:
        """Empty preprocess method

        This method is defined because PyTorch multiprocessing can't pickle lambdas

        :param _input: Input to preprocess
        :type _input: Any

        :return: Preprocessed input
        :rtype: Any
        """
        return _input
