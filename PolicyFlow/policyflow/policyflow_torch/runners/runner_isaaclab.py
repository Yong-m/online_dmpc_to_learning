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
    make_wandb_cb,
    make_save_model_cb,
)


def _clone_dict(d):
    """Clone a dict of tensors without copy.deepcopy (which fails on non-leaf tensors)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.clone()
        elif isinstance(v, dict):
            out[k] = _clone_dict(v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class IsaaclabRunner:
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
        # Saved obs/env_info between chunked train() calls (avoids re-reset of inference tensors)
        self._train_state: tuple = None
        # Persisted across chunked train() calls so the same directory / callbacks are reused
        self._experiment_dir: str = None
        self._learn_cb: list = None

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
        start_iteration: int = 0,
    ) -> None:
        """Runs a number of learning iterations.
        Args:
            return_epochs (int): The number of epochs over which to aggregate the return. Defaults to 100.
            start_iteration (int): The iteration to start from (for resuming). Defaults to 0.
        """
        max_iterations = self._cfg.get("max_iterations", 0)
        rollouts = self._cfg.get("rollouts", 24)
        # end_iteration: train exactly return_epochs iters from start_iteration
        end_iteration = min(start_iteration + return_epochs, max_iterations)

        if self._learn_cb is None:
            directory = self._cfg.get("log_dir", "")
            experiment_name = self._cfg.get("experiment_name", "")
            if not directory:
                directory = os.path.join(os.getcwd(), "runs")
            if experiment_name:
                directory = os.path.join(directory, experiment_name)
            time_stamp = "{}".format(
                datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f"),
            )
            self._experiment_dir = os.path.join(directory, time_stamp)
            print(f"[INFO] Logging experiment in directory: {self._experiment_dir}")
            self._learn_cb = []
            if "save_interval" in self._cfg:
                self._learn_cb.append(
                    make_interval_cb(
                        make_save_model_cb(self._experiment_dir),
                        self._cfg["save_interval"],
                    )
                )
            self._learn_cb.append(make_tensorboard_cb(self._experiment_dir))
            if self._cfg.get("wandb_project"):
                self._learn_cb.append(make_wandb_cb(
                    project=self._cfg["wandb_project"],
                    experiment_name=experiment_name or time_stamp,
                    config=self._cfg,
                ))
        learn_cb = self._learn_cb

        episode_statistics = {
            "current_iteration": 0,
            "info": [],
            "lengths": [],
            "returns": [],
            "training_info": {},
        }
        current_episode_lengths = torch.zeros(self._env.num_envs, dtype=torch.float)
        current_cumulative_rewards = torch.zeros(self._env.num_envs, dtype=torch.float)

        if self._train_state is not None:
            # Chunked training: resume from saved obs/env_info without re-resetting.
            # Re-resetting would fail because step() inside inference_mode() leaves
            # physics-state tensors (e.g. root_link_pose_w) as inference tensors, and
            # env.reset() tries to do an inplace update on them outside inference_mode.
            obs_dict, env_info = self._train_state
        else:
            # Fresh start: reset env normally.
            obs_dict, env_info = self._env.reset()
            self._env.episode_length_buf = torch.randint_like(
                self._env.episode_length_buf, high=int(self._env.max_episode_length)
            )

        # if start_iteration > 0:
        #     print(f"[INFO] Resuming training from iteration {start_iteration}")

        for _iter in tqdm.tqdm(
            range(start_iteration, end_iteration),
            initial=start_iteration,
            total=max_iterations,
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
                        observations_dict=_clone_dict(obs_dict),
                        environement_info=_clone_dict(env_info),
                        actions=actions.clone(),
                        rewards=rewards.clone(),
                        next_observations_dict=_clone_dict(next_obs_dict),
                        dones=dones.clone(),
                        actions_info=_clone_dict(actions_info),
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

        # Save obs/env_info for the next chunked train() call.
        # obs_dict here contains inference tensors (from the last step inside inference_mode),
        # but that is fine: the next call will use them directly inside inference_mode too.
        self._train_state = (obs_dict, env_info)

    def load(self, path: str) -> int:
        """Restores the agent and runner state from a file.

        Returns:
            The iteration number stored in the checkpoint (0 if not found).
        """
        content = torch.load(path, map_location=self._device, weights_only=False)
        if "model_state_dicts" in content:
            # New format: load each model in-place so optimizer param references stay valid.
            for k, sd in content["model_state_dicts"].items():
                self._model_load_state_dict(self._agent.model_dict[k], sd)
            self._optimizer_load_state_dict(content.get("optimizer"))
        else:
            # Legacy format fallback. Older checkpoints serialized model objects inside
            # agent["model_dict"]; assigning that dict would break optimizer parameter
            # references, so load those weights into the existing modules in-place.
            assert "agent" in content
            agent_state = content["agent"]
            legacy_models = agent_state.get("model_dict")
            if legacy_models is None:
                raise KeyError('Legacy checkpoint is missing agent["model_dict"].')
            for k, loaded_model in legacy_models.items():
                if k not in self._agent.model_dict:
                    print(f"[WARN] Skipping unknown legacy model key during resume: {k}")
                    continue
                self._model_load_state_dict(
                    self._agent.model_dict[k],
                    self._model_state_dict(loaded_model),
                )
            self._optimizer_load_state_dict(agent_state.get("optimizer"))
        iteration = content.get("iteration", 0)
        # Fallback: parse iteration from filename (model_500.pt -> 500)
        if iteration == 0:
            import re
            match = re.search(r"model_(\d+)\.pt", os.path.basename(path))
            if match:
                iteration = int(match.group(1))
        print(f"[INFO] Loaded checkpoint from {path} (iteration {iteration})")
        return iteration

    @staticmethod
    def _model_state_dict(model):
        """Return a serializable state dict regardless of model type."""
        if isinstance(model, torch.nn.Module):
            return {"_type": "nn_module", "state": model.state_dict()}
        # ContinuousNormalizingFlow and similar: save sub-modules directly.
        sub = {}
        for attr in ("model", "model_ema", "model_last", "model_proximal"):
            m = getattr(model, attr, None)
            if m is not None and isinstance(m, torch.nn.Module):
                sub[attr] = m.state_dict()
        return {"_type": "flow", "state": sub}

    @staticmethod
    def _model_load_state_dict(model, sd):
        """Load state dict in-place, preserving optimizer param references."""
        if sd["_type"] == "nn_module":
            model.load_state_dict(sd["state"])
        else:
            for attr, state in sd["state"].items():
                getattr(model, attr).load_state_dict(state)

    def _optimizer_load_state_dict(self, state):
        """Load optimizer state when compatible; keep fresh optimizer otherwise."""
        if state is None:
            print("[WARN] Checkpoint has no optimizer state; using fresh optimizer.")
            return
        try:
            self._agent.optimizer.load_state_dict(state)
        except ValueError as exc:
            print(f"[WARN] Optimizer state incompatible with current model; using fresh optimizer. ({exc})")

    def save(self, path: str, data: Any = None, iteration: int = 0) -> None:
        """Saves the agent and runner state to a file."""
        content = {
            "model_state_dicts": {
                k: self._model_state_dict(v)
                for k, v in self._agent.model_dict.items()
            },
            "optimizer": self._agent.optimizer.state_dict(),
            "iteration": iteration,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(content, path)

    def export_onnx(self, path: str) -> None:
        """Exports the agent's policy network to ONNX format."""
        model, args, kwargs = self._agent.export_onnx()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.onnx.export(model, args, path, **kwargs)
