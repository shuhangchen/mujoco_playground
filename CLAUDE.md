# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation & Setup
```bash
# Create virtual environment
uv venv --python 3.12
source .venv/bin/activate

# Install CUDA 12 JAX first
uv pip install -U "jax[cuda12]==0.6.0"

# Install playground with all dependencies
uv pip install -e ".[all]"

# Verify installation
python -c "import mujoco_playground"
```

### Code Quality & Testing
```bash
# Install development dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Run all pre-commit checks
pre-commit run --all-files

# Manual linting (alternative to pre-commit)
pyink .
isort .
pylint . --rcfile=pylintrc
pytype .

# Run tests
pytest
```

### Training Commands
```bash
# Train PPO agent on specific environment
python learning/train_jax_ppo.py --env_name=go2_walk --num_timesteps=1000000

# Train with RSL-RL
python learning/train_rsl_rl.py --env_name=go2_walk
```

## Architecture Overview

### Core Components

**Environment Registry (`mujoco_playground/_src/registry.py`)**
- Central registry for all environments across three suites: dm_control_suite, locomotion, manipulation
- Provides `load()` function to instantiate environments with configs
- Handles default configuration loading per environment

**Base Environment (`mujoco_playground/_src/mjx_env.py`)**
- Abstract base class `MjxEnv` for all MuJoCo MJX environments
- Handles asset loading, external dependencies (mujoco_menagerie), and XML parsing
- Provides common interface for reset, step, and observation functions

**Environment Suites**
- `dm_control_suite/`: Classic control tasks (cartpole, acrobot, etc.)
- `locomotion/`: Quadruped/bipedal robots (Go1/Go2, H1, G1, etc.)
- `manipulation/`: Robotic manipulation (Franka Panda, ALOHA, Leap Hand)

**Wrappers (`mujoco_playground/_src/wrapper.py`)**
- Environment wrappers for different RL frameworks
- Handles observation/action space transformations
- Includes PyTorch wrapper for non-JAX frameworks

### Training Integration

**Configuration System**
- Environment-specific hyperparameters in `mujoco_playground/config/`
- Separate config files for each suite (dm_control_suite_params.py, etc.)
- Uses ml_collections for structured configuration

**Training Scripts**
- `learning/train_jax_ppo.py`: JAX-based PPO training with Brax
- `learning/train_rsl_rl.py`: Integration with RSL-RL framework
- Supports both vision and state-based training

## Key Implementation Details

### GPU Precision
For NVIDIA Ampere GPUs (RTX 30/40 series), export `JAX_DEFAULT_MATMUL_PRECISION=highest` to avoid reproducibility issues with TF32.

### External Dependencies
- Automatically downloads mujoco_menagerie assets on first import
- XML files stored in `mujoco_playground/_src/*/xmls/`
- Asset loading handled through etils.epath for cross-platform compatibility

### Environment Loading Pattern
```python
import mujoco_playground
env = mujoco_playground.load("go2_walk")
```

### Common GPU Issues
If encountering "cusolver internal error", run:
```bash
unset CUDA_PATH && unset LD_LIBRARY_PATH
```

## Testing Strategy

Tests are located in `mujoco_playground/_src/` with `*_test.py` naming convention. Key test files:
- `registry_test.py`: Environment loading and registration
- `wrapper_test.py`: Environment wrapper functionality
- `dm_control_suite_test.py`: Classic control suite validation
- `locomotion_test.py`: Locomotion environment tests
- `manipulation_test.py`: Manipulation environment tests