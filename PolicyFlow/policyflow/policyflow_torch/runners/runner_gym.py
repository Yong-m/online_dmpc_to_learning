from __future__ import annotations
import os
import copy
import torch
import sys
import datetime
import tqdm
from typing import Any, Optional
from policyflow_torch.env import Wrapper
from policyflow_torch.agents import Agent
from policyflow_torch.runners.callbacks import (
    make_interval_cb,
    make_tensorboard_cb,
    make_save_model_cb,
)


class GymRunner:
    def __init__(
        self,
        env: Wrapper,
        agent: Agent,
        cfg: Optional[dict] = None,
    ) -> None:
        """
        Args:
            environment (rsl_rl.env.VecEnv): The environment to run the agent in.
            agent (rsl_rl.algorithms.agent): The RL agent to run.
            device (str): The device to run on.
        """
        self._env = env
        self._agent = agent
        self._device = env.device
        self._cfg = cfg

    def evaluate(self, steps: int, return_epochs: int = 100, log=False) -> float:
        """Evaluates the agent for a number of steps.

        Args:
            steps (int): The number of steps to evaluate the agent for.
            return_epochs (int): The number of epochs over which to aggregate the return. Defaults to 100.
        """

        if log:
            eval_cb = []
            directory = self._cfg.get("log_dir", "")
            experiment_name = self._cfg.get("experiment_name", "")
            if not directory:
                directory = os.path.join(os.getcwd(), "runs")
            if experiment_name:
                directory = os.path.join(directory, experiment_name)
            time_stamp = "{}".format(
                datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f"),
            )
            experiment_dir = os.path.join(directory, time_stamp)

            if experiment_dir:
                print(f"[INFO] Logging experiment in directory: {experiment_dir}")
                eval_cb.append(make_tensorboard_cb(experiment_dir))

        episode_statistics = {
            "current_iteration": 0,
            "info": [],
            "lengths": [],
            "returns": [],
        }
        current_cumulative_rewards = torch.zeros(self._env.num_envs, dtype=torch.float)
        current_episode_lengths = torch.zeros(self._env.num_envs, dtype=torch.int)

        policy = self.get_inference_policy()
        obs_dict, env_info = self._env.reset()

        with torch.inference_mode():
            for _ in tqdm.tqdm(
                range(steps), disable=False, file=sys.stdout, desc="step"
            ):
                actions = policy(obs_dict)
                # step the environments
                next_obs_dict, rewards, dones, env_info = self._env.step(actions)
                obs_dict = next_obs_dict

                # Gather statistics
                if "log" in env_info:
                    episode_statistics["info"].append(env_info["log"])
                dones_idx = (dones + env_info["time_outs"]).nonzero().cpu()
                current_cumulative_rewards += rewards.cpu()
                current_episode_lengths += 1
                completed_lengths = current_episode_lengths[dones_idx][:, 0].cpu()
                completed_returns = current_cumulative_rewards[dones_idx][:, 0].cpu()
                current_episode_lengths[dones_idx] = 0.0
                current_cumulative_rewards[dones_idx] = 0.0
                episode_statistics["lengths"].extend(completed_lengths.tolist())
                episode_statistics["returns"].extend(completed_returns.tolist())
                episode_statistics["lengths"] = episode_statistics["lengths"][
                    -return_epochs:
                ]
                episode_statistics["returns"] = episode_statistics["returns"][
                    -return_epochs:
                ]
            if log:
                for cb in eval_cb:
                    cb(self, episode_statistics)

    def get_inference_policy(self, device=None):
        return self._agent.get_inference_policy(device)

    def train(
        self,
        return_epochs: int = 100,
    ) -> None:
        """Runs a number of learning iterations.
        Args:
            return_epochs (int): The number of epochs over which to aggregate the return. Defaults to 100.
        """
        max_iterations = self._cfg.get("max_iterations", 0)
        rollouts = self._cfg.get("rollouts", 24)

        directory = self._cfg.get("log_dir", "")
        experiment_name = self._cfg.get("experiment_name", "")
        if not directory:
            directory = os.path.join(os.getcwd(), "runs")
        if experiment_name:
            directory = os.path.join(directory, experiment_name)
        time_stamp = "{}".format(
            datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f"),
        )
        experiment_dir = os.path.join(directory, time_stamp)
        learn_cb = []
        print(f"[INFO] Logging experiment in directory: {experiment_dir}")
        if "save_interval" in self._cfg:
            learn_cb.append(
                make_interval_cb(
                    make_save_model_cb(experiment_dir),
                    self._cfg["save_interval"],
                )
            )
        learn_cb.append(make_tensorboard_cb(experiment_dir))

        episode_statistics = {
            "current_iteration": 0,
            "info": [],
            "lengths": [],
            "returns": [],
            "training_info": {},
        }
        current_episode_lengths = torch.zeros(self._env.num_envs, dtype=torch.float)
        current_cumulative_rewards = torch.zeros(self._env.num_envs, dtype=torch.float)

        # reset env
        obs_dict, env_info = self._env.reset()

        for _iter in tqdm.tqdm(
            range(max_iterations),
            disable=False,
            file=sys.stdout,
            desc="iter",
            leave=False,
        ):
            for _ in tqdm.tqdm(
                range(rollouts),
                disable=False,
                file=sys.stdout,
                desc="step",
                leave=False,
            ):
                with torch.inference_mode():
                    # compute actions
                    actions, actions_info = self._agent.draw_actions(obs_dict, env_info)
                    # step the environments
                    next_obs_dict, rewards, dones, env_info = self._env.step(actions)
                    # record the environments' transitions
                    self._agent.process_transition(
                        observations_dict=copy.deepcopy(obs_dict),
                        environement_info=copy.deepcopy(env_info),
                        actions=actions.clone(),
                        rewards=rewards.clone(),
                        next_observations_dict=copy.deepcopy(next_obs_dict),
                        dones=dones.clone(),
                        actions_info=copy.deepcopy(actions_info),
                    )
                    obs_dict = next_obs_dict

                    # Gather statistics
                    if "log" in env_info:
                        episode_statistics["info"].append(env_info["log"])
                    dones_idx = (dones + env_info["time_outs"]).nonzero().cpu()
                    current_cumulative_rewards += rewards.cpu()
                    current_episode_lengths += 1
                    completed_lengths = current_episode_lengths[dones_idx][:, 0].cpu()
                    completed_returns = current_cumulative_rewards[dones_idx][
                        :, 0
                    ].cpu()
                    current_episode_lengths[dones_idx] = 0.0
                    current_cumulative_rewards[dones_idx] = 0.0
                    episode_statistics["lengths"].extend(completed_lengths.tolist())
                    episode_statistics["returns"].extend(completed_returns.tolist())
                    episode_statistics["lengths"] = episode_statistics["lengths"][
                        -return_epochs:
                    ]
                    episode_statistics["returns"] = episode_statistics["returns"][
                        -return_epochs:
                    ]

            episode_statistics["training_info"] = self._agent.update()
            episode_statistics["current_iteration"] = _iter
            terminate = False
            for cb in learn_cb:
                terminate = (cb(self, episode_statistics) == False) or terminate
            if terminate:
                break
            episode_statistics["info"].clear()

    def load(self, path: str) -> Any:
        """Restores the agent and runner state from a file."""
        content = torch.load(path, map_location=self._device, weights_only=False)
        assert "agent" in content
        self._agent.load_state_dict(content["agent"])

    def save(self, path: str, data: Any = None) -> None:
        """Saves the agent and runner state to a file."""
        content = {
            "agent": self._agent.state_dict(),
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(content, path)

    def export_onnx(self, path: str) -> None:
        """Exports the agent's policy network to ONNX format."""
        model, args, kwargs = self._agent.export_onnx()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.onnx.export(model, args, path, **kwargs)
