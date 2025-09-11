# Contractive Diffusion Policies (CDP)

Implementation of contractive reverse diffusion process to improve policies for `offline policy learning`. CDP aims to improve diffusion policy's sample efficiency and general performance by enforcing contraction constraints on the learned policy through Jacobian/eigenvalue regularization.

---
## Overview

Diffusion policies are widely-used generative models in robot learning often formulated by SDE. However, the same SDE modeling that provides diffusion policies with the flexibility to learn diverse behavior suffers from discretization and integration errors, requires large datasets for precise score matching, and experiences inconsistencies in action generation.

We propose SDE by promoting contraction in the reverse diffusion process to mitigate solver errors, and to reduces unwanted action variance during the sampling process.


`CDP can be implemented by minimal modifications to ANY diffusion policy framework.`

```python
# Assuming a defined diffusion backbone during training
from cleandiffuser.nn_diffusion import ChiUNet1d
nn_diffusion: ChiUNet1d

# Calculate the score jacobian
# Our theory shows this is enough for enforcing contraction throughout the reverse diffusion process
score_jacobian = jacobian(nn_diffusion, action)

# Find the largest eigenvalue of the symmetric part
largest_eigenval = approx_largest_eigenval(sym(score_jacobian))

# Penalize and add it to the loss (simplistic version)
contraction_loss = ctr_coeff * relu((largest_eigenval + ctr_th) ** 2)
total_loss = diffusion_loss + contraction_loss
```

### Benefits of contractive sampling process
![CDP Concept](media/cdp_concept.png)
![CDP Steps](media/cdp_training.png)

---

## Setup Guide

### 1. Environment Setup

Create and activate a Conda environment:

```bash
# Create Python 3.9 environment
conda create -n cdp python=3.9

# Activate the environment
conda activate cdp
```

### 2. Install Clean Diffuser

This project relies on the [Clean Diffuser](https://github.com/CleanDiffuserTeam/CleanDiffuser) implementation for diffusion and conditioning architectures.

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
# Install Mujoco-py dependencies, remove libgl1-mesa-glx for Ubuntu 24.04
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

### 4. Install Robomimic

Note: There exists an incompatiblity between MuJoCu for Robosuite and D4RL. Easiest fix for now is by cloning the previous conda environment or downgrading the dm_control package.

```bash
conda create --name contractive-diffuser-robomimic --clone contractive-diffuser
```

And now install Robomimic:

```bash
# Install Robomimic from source (recommended)
cd <PATH_TO_ROBOMIMIC_INSTALL_DIR>
git clone https://github.com/ARISE-Initiative/robomimic.git
cd robomimic
pip install -e .

# And Robosuite!
cd <PATH_TO_ROBOSUITE_INSTALL_DIR>
git clone https://github.com/ARISE-Initiative/robosuite.git
cd robosuite
pip install -e .
```

## Simple runs
Running CDP on a selected D4RL task is just as simple as running the [edp_d4rl.py](edp_d4rl.py) file with proper arguments.

```bash
python edp_d4rl.py env_name="kitchen" task="kitchen-complete-v0" loss_type="jacobian" loss_weights.jacobian=1.0
```

And for running CDP in an imitation learning setup on robomimic dataset, try

```python
python dbc_robomimic.py task="$task" loss_type="jacobian" loss_weights.jacobian=0.1

```

## All Experiments

To run all experiments for all subtasks of a certain benchmark, use either [il_exps.bash](il_exps.bash) for imitation learning or [offline_rl_exps.bash](offline_rl_exps.bash) in the following way.

```bash
chmod +x il_exps.bash
./il_exps.bash <robomimic_environment> <seeds, default=5>

# e.g., single run of all experiments on low-dim robomimic lift, square, transport, and can tasks
./il_exps.bash robomimic_lowdim 1
```

```bash
chmod +x offline_rl_exps.bash
./offline_rl_exps.bash <d4rl_environment> <seeds, default=5>

# e.g., single run of all experiments on low-dim robomimic lift, square, transport, and can tasks
./offline_rl_exps.bash robomimic_lowdim 1
```

### Background processes

The script launches background jobs for better parallelization.
To kill these background processes, you can use ```pkill``` or just ```kill```. For instance ```pkill -9 -f kitchen``` for kitchen training processes or ```kill -9 <pid>``` if you have a specific process Id. Check list of processes with ```ps -aux | grep <part_of_env_name>```.

---

## Hydra Configurations

All experiments are managed via [Hydra](https://hydra.cc/). Configuration files can be found in the [configs](configs) directory.

---

## Contribution

We welcome contributions that improve our work! Please open an issue or submit a pull request.

---

## Authors

Anonymous authors.
