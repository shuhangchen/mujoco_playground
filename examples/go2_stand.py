import os
import subprocess


# Configure MuJoCo to use the EGL rendering backend (requires GPU)
print('Setting environment variable to use GPU rendering:')
os.environ['MUJOCO_GL'] = 'egl'

try:
  print('Checking that the installation succeeded:')
  import mujoco

  mujoco.MjModel.from_xml_string('<mujoco/>')
except Exception as e:
  raise e from RuntimeError(
      'Something went wrong during installation. Check the shell output above '
      'for more information.\n'
      'If using a hosted Colab runtime, make sure you enable GPU acceleration '
      'by going to the Runtime menu and selecting "Choose runtime type".'
  )

print('Installation successful.')

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

import json
import itertools
import time
from typing import Callable, List, NamedTuple, Optional, Union
import numpy as np

# Graphics and plotting.
import mediapy as media
import matplotlib.pyplot as plt

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

from datetime import datetime
import functools
import os
from typing import Any, Dict, Sequence, Tuple, Union
from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.io import html, mjcf, model
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac
from etils import epath
from flax import struct
from flax.training import orbax_utils
from IPython.display import HTML, clear_output
import jax
from jax import numpy as jp
from matplotlib import pyplot as plt
import mediapy as media
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np
from orbax import checkpoint as ocp

from mujoco_playground import wrapper
from mujoco_playground import registry

print(f"locomotion env {registry.locomotion.ALL_ENVS}")

# hand stand policy
from mujoco_playground.config import locomotion_params

env_name = 'Go2Handstand'
env = registry.load(env_name)
env_cfg = registry.get_default_config(env_name)
ppo_params = locomotion_params.brax_ppo_config(env_name)

ckpt_path = epath.Path("checkpoints").resolve() / env_name
ckpt_path.mkdir(parents=True, exist_ok=True)
print(f"{ckpt_path}")

with open(ckpt_path / "config.json", "w") as fp:
  json.dump(env_cfg.to_dict(), fp, indent=4)


#@title Training fn definition
x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]


def policy_params_fn(current_step, make_policy, params):
  del make_policy  # Unused.
  orbax_checkpointer = ocp.PyTreeCheckpointer()
  save_args = orbax_utils.save_args_from_target(params)
  path = ckpt_path / f"{current_step}"
  orbax_checkpointer.save(path, params, force=True, save_args=save_args)

plt.ion()
plt.figure(figsize=(10, 6))

def progress(num_steps, metrics):
  clear_output(wait=True)

  times.append(datetime.now())
  x_data.append(num_steps)
  y_data.append(metrics["eval/episode_reward"])
  y_dataerr.append(metrics["eval/episode_reward_std"])

  plt.xlim([0, ppo_params["num_timesteps"] * 1.25])
  plt.xlabel("# environment steps")
  plt.ylabel("reward per episode")
  plt.title(f"y={y_data[-1]:.3f}")
  plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")
  plt.draw()
  plt.pause(0.001)

randomizer = registry.get_domain_randomizer(env_name)
ppo_training_params = dict(ppo_params)
network_factory = ppo_networks.make_ppo_networks
if "network_factory" in ppo_params:
  del ppo_training_params["network_factory"]
  network_factory = functools.partial(
      ppo_networks.make_ppo_networks,
      **ppo_params.network_factory
  )

train_fn = functools.partial(
    ppo.train, **dict(ppo_training_params),
    network_factory=network_factory,
    randomization_fn=randomizer,
    progress_fn=progress,
    policy_params_fn=policy_params_fn,
)


make_inference_fn, params, metrics = train_fn(
    environment=registry.load(env_name, config=env_cfg),
    eval_env=registry.load(env_name, config=env_cfg),
    wrap_env_fn=wrapper.wrap_for_brax_training,
)
print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")

#@title Rollout and Render
inference_fn = make_inference_fn(params, deterministic=True)
jit_inference_fn = jax.jit(inference_fn)

eval_env = registry.load(env_name, config=env_cfg)
jit_reset = jax.jit(eval_env.reset)
jit_step = jax.jit(eval_env.step)

rng = jax.random.PRNGKey(12345)
rollout = []
rewards = []
torso_height = []
actions = []
torques = []
power = []
qfrc_constraint = []
qvels = []
power1 = []
power2 = []
for _ in range(10):
  rng, reset_rng = jax.random.split(rng)
  state = jit_reset(reset_rng)
  for i in range(env_cfg.episode_length // 2):
    act_rng, rng = jax.random.split(rng)
    ctrl, _ = jit_inference_fn(state.obs, act_rng)
    actions.append(ctrl)
    state = jit_step(state, ctrl)
    rollout.append(state)
    rewards.append(
        {k[7:]: v for k, v in state.metrics.items() if k.startswith("reward/")}
    )
    torso_height.append(state.data.qpos[2])
    torques.append(state.data.actuator_force)
    qvel = state.data.qvel[6:]
    power.append(jp.sum(jp.abs(qvel * state.data.actuator_force)))
    qfrc_constraint.append(jp.linalg.norm(state.data.qfrc_constraint[6:]))
    qvels.append(jp.max(jp.abs(qvel)))
    frc = state.data.actuator_force
    qvel = state.data.qvel[6:]
    power1.append(jp.sum(frc * qvel))
    power2.append(jp.sum(jp.abs(frc * qvel)))


render_every = 2
fps = 1.0 / eval_env.dt / render_every
traj = rollout[::render_every]

scene_option = mujoco.MjvOption()
scene_option.geomgroup[2] = True
scene_option.geomgroup[3] = False
scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False

frames = eval_env.render(
    traj, camera="side", scene_option=scene_option, height=480, width=640
)
media.write_video("go2_stand_rollout.mp4", frames, fps=fps)
print("Video saved as go2_stand_rollout.mp4")

power = jp.array(power1)
print(f"Max power: {jp.max(power)}")


# finetune the previous checkpoint
env_cfg = registry.get_default_config(env_name)
env_cfg.energy_termination_threshold = 400  # lower energy termination threshold
env_cfg.reward_config.energy = -0.003  # non-zero negative `energy` reward
env_cfg.reward_config.dof_acc = -2.5e-7  # non-zero negative `dof_acc` reward

FINETUNE_PATH = epath.Path(ckpt_path)
latest_ckpts = list(FINETUNE_PATH.glob("*"))
latest_ckpts = [ckpt for ckpt in latest_ckpts if ckpt.is_dir()]
latest_ckpts.sort(key=lambda x: int(x.name))
latest_ckpt = latest_ckpts[-1]
restore_checkpoint_path = latest_ckpt


x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]

make_inference_fn, params, metrics = train_fn(
    environment=registry.load(env_name, config=env_cfg),
    eval_env=registry.load(env_name, config=env_cfg),
    wrap_env_fn=wrapper.wrap_for_brax_training,
    restore_checkpoint_path=restore_checkpoint_path,  # restore from the checkpoint!
    seed=1,
)
print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")


#@title Rollout and Render Finetune Policy
inference_fn = make_inference_fn(params, deterministic=True)
jit_inference_fn = jax.jit(inference_fn)

eval_env = registry.load(env_name, config=env_cfg)
jit_reset = jax.jit(eval_env.reset)
jit_step = jax.jit(eval_env.step)

rng = jax.random.PRNGKey(12345)
rollout = []
rewards = []
torso_height = []
actions = []
torques = []
power = []
qfrc_constraint = []
qvels = []
power1 = []
power2 = []
for _ in range(10):
  rng, reset_rng = jax.random.split(rng)
  state = jit_reset(reset_rng)
  for i in range(env_cfg.episode_length // 2):
    act_rng, rng = jax.random.split(rng)
    ctrl, _ = jit_inference_fn(state.obs, act_rng)
    actions.append(ctrl)
    state = jit_step(state, ctrl)
    rollout.append(state)
    rewards.append(
        {k[7:]: v for k, v in state.metrics.items() if k.startswith("reward/")}
    )
    torso_height.append(state.data.qpos[2])
    torques.append(state.data.actuator_force)
    qvel = state.data.qvel[6:]
    power.append(jp.sum(jp.abs(qvel * state.data.actuator_force)))
    qfrc_constraint.append(jp.linalg.norm(state.data.qfrc_constraint[6:]))
    qvels.append(jp.max(jp.abs(qvel)))
    frc = state.data.actuator_force
    qvel = state.data.qvel[6:]
    power1.append(jp.sum(frc * qvel))
    power2.append(jp.sum(jp.abs(frc * qvel)))


render_every = 2
fps = 1.0 / eval_env.dt / render_every
traj = rollout[::render_every]

scene_option = mujoco.MjvOption()
scene_option.geomgroup[2] = True
scene_option.geomgroup[3] = False
scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False

frames = eval_env.render(
    traj, camera="side", scene_option=scene_option, height=480, width=640
)
media.write_video("go2_stand_rollout_finetune.mp4", frames, fps=fps)
print("Video saved as go2_stand_rollout_finetune.mp4")

power = jp.array(power1)
print(f"Max power: {jp.max(power)}")

plt.ioff()
input("Press Enter to close all plots and exit...")