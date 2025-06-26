# Spot Robot Manipulation System

A complete robotic manipulation system for Boston Dynamics Spot robot that combines interactive perception with reinforcement learning for object manipulation.

## Overview

This system implements a pipeline that:

1. **Interactive Perception**: Uses "wiggling" motions to estimate joint type (prismatic vs revolute) and parameters
2. **State Construction**: Builds state vectors from estimated joint parameters
3. **Policy Execution**: Runs trained PPO agents to perform manipulation tasks

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Spot Robot    │    │ Interactive      │    │ Policy          │
│   Interface     │◄──►│ Perception       │◄──►│ Executor        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ State           │    │ Joint Parameter  │    │ Multi-head PPO  │
│ Constructor     │    │ Estimation       │    │ Agent           │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Components

### 1. Main System (`main.py`)
- Orchestrates the complete manipulation pipeline
- Handles configuration loading and error management
- Provides command-line interface

### 2. Interactive Perception (`interactive_perception.py`)
- Estimates joint type and parameters from trajectory data
- Implements prismatic and revolute joint detection
- Uses PCA and circle fitting for parameter estimation

### 3. State Constructor (`state_constructor.py`)
- Builds state vectors for policy input
- Handles both prismatic and revolute joint states
- Maintains 6-dimensional state representation

### 4. Policy Executor (`policy_executor.py`)
- Loads and executes trained PPO agents
- Implements multi-head actor-critic networks
- Handles action application and success detection

### 5. Spot Interface (`spot_interface.py`)
- Interface to Boston Dynamics Spot robot
- Currently provides mock implementation for testing
- Ready for bosdyn SDK integration

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd spot_manipulation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up configuration:
```bash
cp config/spot_config.json.example config/spot_config.json
# Edit config/spot_config.json with your settings
```

## Usage

### Basic Usage

```bash
python main.py --target_x 0.1 --target_y 0.0 --target_z 0.0
```

### Command Line Options

- `--config`: Path to configuration file (default: `config/spot_config.json`)
- `--target_x`: Target X coordinate
- `--target_y`: Target Y coordinate  
- `--target_z`: Target Z coordinate

### Configuration

Edit `config/spot_config.json` to customize:

- **Spot Robot Settings**: IP address, credentials, movement parameters
- **Policy Settings**: Checkpoint paths, action scaling, success thresholds
- **Interactive Perception**: Wiggle distances, delays, error thresholds
- **Manipulation**: Timeouts, safety margins, max attempts

## Interactive Perception

The system performs interactive perception by:

1. **Wiggling**: Moving the end-effector in 4 directions (forward, backward, left, right)
2. **Trajectory Recording**: Capturing end-effector positions during wiggling
3. **Joint Estimation**: Using the trajectory to estimate:
   - Joint type (prismatic vs revolute)
   - Joint direction/axis
   - Joint position (for revolute joints)
   - Joint radius (for revolute joints)

## Policy Execution

The system executes trained PPO policies by:

1. **State Construction**: Building state vectors from joint parameters
2. **Action Generation**: Using multi-head PPO to generate actions
3. **Action Application**: Applying dx, dy, dz movements to end-effector
4. **Success Detection**: Monitoring task completion

## Training

To train new policies, you'll need to:

1. Set up training environments similar to the original FLEX system
2. Train separate policies for prismatic and revolute joints
3. Save checkpoints in the format expected by the policy executor

## Testing

The system includes a mock Spot interface for testing without actual hardware:

```python
from spot_interface import MockSpotInterface

config = {'use_mock': True}
spot = MockSpotInterface(config)
```

## Integration with Real Spot Robot

To integrate with actual Spot hardware:

1. Install bosdyn SDK:
```bash
pip install bosdyn-client
```

2. Update `spot_interface.py` with actual bosdyn SDK calls
3. Configure robot IP and credentials in config file
4. Test with small movements first

## File Structure

```
spot_manipulation/
├── main.py                 # Main entry point
├── interactive_perception.py  # Joint estimation
├── state_constructor.py    # State vector construction
├── policy_executor.py      # PPO policy execution
├── spot_interface.py       # Spot robot interface
├── config/
│   └── spot_config.json    # Configuration file
├── checkpoints/            # Trained policy checkpoints
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Dependencies

- **numpy**: Numerical computations
- **torch**: Deep learning (PPO networks)
- **scikit-learn**: Machine learning (PCA, linear regression)
- **scipy**: Scientific computing (optimization)
- **matplotlib**: Plotting (optional)
- **bosdyn-client**: Spot robot SDK (when implementing real interface)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Acknowledgments

This system is based on the FLEX robotic manipulation framework and adapts its interactive perception and reinforcement learning components for use with Boston Dynamics Spot robots. 