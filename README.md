
# FLEX: Framework for Learning Robot-Agnostic Force-based Skills Involving Sustained Contact Object Manipulation


This repository contains the codebase for FLEX, a framework for learning force-based reinforcement learning (RL) skills that are robot-agnostic and generalizable to various objects with similar joint configurations. Below, you will find instructions on how to install the dependencies and run FLEX. If you have any questions or wish to contribute or connect, please [contact the authors](#connect).

## Table of Contents
- [FLEX: Framework for Learning Robot-Agnostic Force-based Skills Involving Sustained Contact Object Manipulation](#flex-framework-for-learning-robot-agnostic-force-based-skills-involving-sustained-contact-object-manipulation)
  - [Table of Contents](#table-of-contents)
  - [Clone the Repository](#clone-the-repository)
    - [Clone the AO-GRASP Repository](#clone-the-ao-grasp-repository)
  - [Dependencies](#dependencies)
    - [Setting Up the FLEX Environment](#setting-up-the-flex-environment)
  - [Installing the Packages](#installing-the-packages)
  - [Fixing Rendering Issues](#fixing-rendering-issues)
  - [Try It Out!](#try-it-out)
  - [Connect](#connect)

## Clone the Repository

Clone the repository and its submodules (FLEX includes a submodule called `contact_graspnet` for generating grasp proposals). Please ensure that you **only clone the main branch**; the `website` branch is for the project website.

```bash
git clone --recurse-submodules https://github.com/tufts-ai-robotics-group/FLEX.git --single-branch
```

### Clone the AO-GRASP Repository

Clone the AO-GRASP repository into a separate directory. **Do not** clone their version of `contact_graspnet`, as it is known to have compatibility issues with newer GPUs (RTX 30 series onwards).

## Dependencies

There are two environments to set up to run FLEX: the main FLEX environment and a separate environment for `contact_graspnet` due to different CUDA versions.

### Setting Up the FLEX Environment

1. **Install Conda**: We recommend Miniconda. Follow the [Miniconda installation instructions](https://docs.anaconda.com/miniconda/miniconda-install/).
2. **Create the FLEX Environment**:
   ```bash
   conda env create --name flex --file=environments.yml
   ```
3. **Create the `contact_graspnet` Environment**: `contact_graspnet` relies on a different version of CUDA, so a separate environment is required.
   ```bash
   cd flex/grasps/aograsp/contact_graspnet/
   conda env create --name cgn --file cgn_env_rtx30.yml
   ```
   > **Note:** The environment **must** be named `cgn` because a script automatically launches this environment when running the grasp module.

## Installing the Packages

1. **Install AO-GRASP**: Navigate to the root directory of the cloned AO-GRASP package and run:
   ```bash
   conda activate flex
   pip install -e .
   ```
   You do not need to reinstall AO-GRASP's dependencies, as they are included in the `flex` environment's requirements.

2. **Install PointNet++ for AO-GRASP**:
   ```bash
   cd aograsp/models/Pointnet2_PyTorch/
   pip install -e .
   cd pointnet2_ops_lib/
   pip install -e .
   ```

3. **Install FLEX**: Navigate to the root directory of the cloned FLEX package and run:
   ```bash
   pip install -e .
   ```

## Fixing Rendering Issues

If you encounter rendering issues with Robosuite (the robot simulator used in this project), set the following environment variable before running the code:

```bash
export MUJOCO_GL="osmesa"
```

To make this change permanent, add the above line to your `~/.bashrc` file.

## Try It Out!

You can test the trained policies or train a new policy using the commands below:

- **Test the Trained Policies**:
   ```bash
   conda activate flex
   cd flex/
   python scripts/parallel_test.py
   ```

- **Train the Policy**:
   ```bash
   conda activate flex
   cd flex/
   python scripts/parallel_train.py
   ```

## Connect

For any questions regarding the paper or the code, please contact us at:
- [shijie.fang@tufts.edu](mailto:shijie.fang@tufts.edu)
- [wenchang.gao@tufts.edu](mailto:wenchang.gao@tufts.edu)
- [shivam.goel@tufts.edu](mailto:shivam.goel@tufts.edu)

