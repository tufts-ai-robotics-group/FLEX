# FLEX:  Framework for Learning Robot-Agnostic Force-based Skills Involving Sustained Contact Object Manipulation 

# Clone the repo
Clone the repo by running \(please only clone main branch, the ```website``` branch is for our project website.\)
FLEX includes a submodule -- contact_graspnet, for generating grasp proposals. Please clone the submodule as well \(```--recurse-submodules```\).
```
git clone --recurse-submodules https://github.com/tufts-ai-robotics-group/FLEX.git --single-branch
```

# Clone the AO-GRASP repo
Clone the AO-GRASP repo to another directory. Please **do not** clone their version of the contact_graspnet. Their version is buggy when using newer GPU versions\(RTX 30 series onwards\).
 
# Depandencies
There are two environments to be installed in order to run FLEX. 
## The FLEX environment
1. Install Conda; here, we recommend Miniconda.
   Please follow the instructions on their website for installing Miniconda. [Link](https://docs.anaconda.com/miniconda/miniconda-install/)
2. Create the main FLEX environment by running
   ```
   conda env create --name flex --file=environments.yml
   ```
3. Create the environment for contact_graspnet. This is because contact_graspnet relies on a different version of CUDA compared to FLEX, thus to environments must by separated.
   ```
   cd flex/grasps/aograsp/contact_graspnet/
   conda env create --name cgn --file cgn_env_rtx30.yml
   ```
   Note that this environment **MUST** be named as ```cgn```, since there will be a script to launch this environment when running grasp module.

# Installing the packages. 
1. Install AO-GRASP by going into the root directory of the cloned ao-grasp package and run. You don't have to reinstall their dependencies because they are already included in the requirements for our ```flex``` environment.
```
conda activate flex
pip install -e .
```
2.  Install PointNet++ for AO-GRASP.
   ```
   cd aograsp/models/Pointnet2_PyTorch/
   pip install -e .
   cd pointnet2_ops_lib/
   pip install -e .
   ```
3. Install FLEX.
   Go into the root directory of the cloned FLEX package and run
   ```
   pip install -e .
   ```

# Fixing issue for rendering

Sometimes Robosuite \(The robot simulator that we use\) faces rendering issues.
Type this in the terminal before running code to get around the OpenGL issue

```export MUJOCO_GL="osmesa"```

To permanently eliminate this issue, add the above line to your ```.bashrc``` file. 

# Try it out!
You can use the trained policies and check their performance by running:
```
conda activate flex
cd flex/
python scripts/parallel_test.py
```

Or, you can train the policy yourself by running:
```
conda activate flex
cd flex/
python scripts/parallel_train.py
```
