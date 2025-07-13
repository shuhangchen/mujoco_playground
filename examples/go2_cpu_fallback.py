import os
import subprocess

# Set JAX to be more conservative with GPU memory
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEMORY_FRACTION'] = '0.5'
os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True --xla_gpu_enable_triton_softmax_fusion=false'

# Try to use GPU, but fall back to CPU if needed
try:
  if subprocess.run('nvidia-smi').returncode:
    raise RuntimeError("No GPU available")
  
  # Add an ICD config so that glvnd can pick up the Nvidia EGL driver.
  NVIDIA_ICD_CONFIG_PATH = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
  if not os.path.exists(NVIDIA_ICD_CONFIG_PATH):
    with open(NVIDIA_ICD_CONFIG_PATH, 'w') as f:
      f.write("""{
      "file_format_version" : "1.0.0",
      "ICD" : {
          "library_path" : "libEGL_nvidia.so.0"
      }
  }
  """)

  # Configure MuJoCo to use the EGL rendering backend (requires GPU)
  print('Setting environment variable to use GPU rendering:')
  os.environ['MUJOCO_GL'] = 'egl'
  
  print('GPU setup successful')
  USE_GPU = True
  
except Exception as e:
  print(f'GPU setup failed: {e}')
  print('Falling back to CPU')
  os.environ['MUJOCO_GL'] = 'osmesa'  # Use software rendering
  USE_GPU = False

try:
  print('Checking that the installation succeeded:')
  import mujoco

  mujoco.MjModel.from_xml_string('<mujoco/>')
except Exception as e:
  raise e from RuntimeError(
      'Something went wrong during installation. Check the shell output above '
      'for more information.'
  )

print('Installation successful.')

import json
import itertools
import time
from typing import Callable, List, NamedTuple, Optional, Union
import numpy as np

# Graphics and plotting.
print("Installing mediapy:")
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

# Configure JAX for better stability
jax.config.update('jax_debug_nans', True)
jax.config.update('jax_debug_infs', True)

# Force JAX to use a specific device if GPU is problematic
if not USE_GPU:
  print("Forcing JAX to use CPU")
  os.environ['CUDA_VISIBLE_DEVICES'] = ''
  jax.config.update('jax_platform_name', 'cpu')

env_name = 'Go2JoystickFlatTerrain'
env = registry.load(env_name)
env_cfg = registry.get_default_config(env_name)

from mujoco_playground.config import locomotion_params
ppo_params = locomotion_params.brax_ppo_config(env_name)

# Reduce memory usage by modifying PPO parameters
print("Reducing memory usage for compatibility...")
ppo_params.num_envs = 1024  # Further reduced
ppo_params.batch_size = 64   # Further reduced
ppo_params.num_timesteps = 10_000_000  # Much shorter for testing
ppo_params.num_evals = 3     # Reduced
ppo_params.unroll_length = 5  # Reduced
ppo_params.num_minibatches = 8  # Reduced

# Use even smaller networks
ppo_params.network_factory.policy_hidden_layer_sizes = (128, 64)  # Much smaller
ppo_params.network_factory.value_hidden_layer_sizes = (128, 64)   # Much smaller

print(f"Modified PPO config:")
print(f"  num_envs: {ppo_params.num_envs}")
print(f"  batch_size: {ppo_params.batch_size}")
print(f"  num_timesteps: {ppo_params.num_timesteps}")
print(f"  network sizes: {ppo_params.network_factory.policy_hidden_layer_sizes}")
print(f"  Using GPU: {USE_GPU}")

registry.get_domain_randomizer(env_name)

x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]


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

  display(plt.gcf())

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
    progress_fn=progress
)

print("Starting training with minimal memory settings...")
try:
  make_inference_fn, params, metrics = train_fn(
      environment=env,
      eval_env=registry.load(env_name, config=env_cfg),
      wrap_env_fn=wrapper.wrap_for_brax_training,
  )
  print(f"time to jit: {times[1] - times[0]}")
  print(f"time to train: {times[-1] - times[1]}")
  print("Training completed successfully!")
except Exception as e:
  print(f"Training failed with error: {e}")
  print("This might be due to GPU memory issues or JAX/CUDA compatibility problems.")
  print("Try running with even smaller parameters or on CPU.") 