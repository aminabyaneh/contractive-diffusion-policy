# Contractive Diffuser

A contractive reverse diffusion process to improve policies for offline reinforcement learning (RL).

---

## Setup Guide

### 1. Environment Setup

Create and activate a Conda or Mamba environment:

```bash
# Create Python 3.9 environment
conda create -n contractive-diffuser python=3.9

# Activate the environment
conda activate contractive-diffuser
```

### 2. Install Clean Diffuser

This project relies on the [Clean Diffuser](https://github.com/CleanDiffuserTeam/CleanDiffuser) implementation for diffusion policies.

```bash
# (Optional) Create a directory for libraries
mkdir libs && cd libs

# Clone the Clean Diffuser repository
git clone https://github.com/CleanDiffuserTeam/CleanDiffuser.git
cd CleanDiffuser

# Install in editable mode
pip install -e .
```

### 3. Install D4RL

We use [D4RL](https://github.com/Farama-Foundation/D4RL) offline datasets for experiments.

```bash
# Install Mujoco-py dependencies
sudo apt-get install libosmesa6-dev libgl1-mesa-glx libglfw3 libglew-dev patchelf

# Navigate to the libs directory if not already there
cd libs

# Clone and install D4RL
git clone https://github.com/Farama-Foundation/D4RL.git
cd D4RL
pip install -e .
```

If you get errors related to Mujoco, you can downgrade the version to 3.1.6 as shown below.

```bash
pip install "dm_control<=1.0.20" "mujoco<=3.1.6"
```

---

## Training Pipelines

Instructions and scripts for training are provided in the [pipelines](pipelines) directory.

---

## Evaluation

See the configs for reproducible results.

---

## Hydra Configurations

All experiments are managed via [Hydra](https://hydra.cc/). Configuration files can be found in the [configs](configs) directory.

---

## Contribution

We welcome contributions that improve our work! Please open an issue or submit a pull request.

---

## Authors

Anonymous authors.
