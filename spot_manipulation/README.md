# Spot Interactive Manipulation System

A robotic manipulation system for Boston Dynamics Spot robot that combines interactive perception with reinforcement learning for object manipulation.

## Overview

This system implements a complete pipeline for autonomous object manipulation:

1. **Object Grasping**: Visual point-and-click object selection and grasping
2. **Interactive Perception**: "Wiggling" motions to estimate joint type (prismatic vs revolute) and parameters
3. **Policy Execution**: TD3 reinforcement learning agents for manipulation tasks

## Architecture

The system consists of four main components:

- **robot.py**: Main system orchestration and Spot robot interface
- **interactive_perception.py**: Joint estimation from trajectory analysis
- **policy_manager.py**: TD3 policy loading and action execution
- **td3.py**: Multi-head TD3 implementation with force direction and magnitude heads

## Key Features

- **Multi-head TD3**: Separate heads for force direction and magnitude as described in manipulation literature
- **Joint-specific policies**: Different trained models for prismatic and revolute joints
- **Interactive perception**: SVD and PCA-based joint parameter estimation
- **State representation**: 6D state vectors combining joint axis and displacement

## Installation

```bash
# Clone repository
git clone <repository-url>
cd spot_manipulation

# Install dependencies
pip install numpy torch scikit-learn scipy bosdyn-client opencv-python
```

## Usage

```bash
python robot.py ROBOT_IP --models-dir models/
```

### Command Line Options

- `hostname`: IP address of Spot robot (required)
- `--models-dir`: Directory containing trained policy models (default: models/)

## Interactive Perception Algorithm

The system estimates joint parameters through trajectory analysis:

1. **Trajectory Collection**: Execute 4-directional wiggling movements (forward, backward, left, right)
2. **Joint Classification**: Compare prismatic vs revolute error metrics
3. **Parameter Estimation**: 
   - **Prismatic**: Extract joint axis using SVD
   - **Revolute**: Fit circle using PCA and least squares optimization

## Policy Architecture

TD3 networks with multi-head actor design:
- **Direction Head**: Outputs normalized force direction vector
- **Magnitude Head**: Outputs force magnitude in [0,1]
- **Final Action**: force = direction × magnitude × scale_factor

## File Structure

```
spot_manipulation/
├── robot.py                    # Main system + Spot interface
├── interactive_perception.py   # Joint estimation + state construction
├── policy_manager.py          # Policy loading + execution
├── td3.py                     # Multi-head TD3 implementation
└── models/                    # Trained policy checkpoints
    ├── prismatic/best_model_actor.pth
    └── revolute/best_model_actor.pth
```

## Configuration

The system uses default parameters optimized for typical manipulation tasks:
- Wiggling distance: 5cm in each direction
- Action scaling: 2cm per policy step
- State dimension: 6 (joint axis + displacement)
- Action dimension: 3 (force vector)

## Training Data Requirements

Policies should be trained with:
- **State space**: [joint_axis_x, joint_axis_y, joint_axis_z, displacement_x, displacement_y, displacement_z]
- **Action space**: [force_x, force_y, force_z] 
- **Separate models**: One for prismatic joints, one for revolute joints

## Dependencies

- **numpy**: Numerical computations
- **torch**: Deep learning framework
- **scikit-learn**: PCA and linear regression
- **scipy**: Optimization algorithms
- **opencv-python**: Image processing for object selection
- **bosdyn-client**: Boston Dynamics Spot SDK

## Safety Notes

- Test with small action scales initially
- Ensure proper emergency stop procedures
- Verify workspace boundaries before operation
- Monitor robot behavior during initial runs

## License

[Add appropriate license]