#!/usr/bin/env python3
"""
Test script for the Spot manipulation system.
Demonstrates the complete pipeline using mock components.
"""

import numpy as np
import json
import os
from spot_interface import MockSpotInterface
from interactive_perception import InteractivePerception
from state_constructor import StateConstructor
from policy_executor import PolicyExecutor


def create_test_trajectory(joint_type='prismatic'):
    """
    Create a test trajectory for interactive perception.
    
    Args:
        joint_type: Type of joint to simulate ('prismatic' or 'revolute')
        
    Returns:
        Trajectory array
    """
    if joint_type == 'prismatic':
        # Create a linear trajectory with some noise
        t = np.linspace(0, 1, 20)
        trajectory = np.zeros((20, 3))
        trajectory[:, 0] = t + np.random.normal(0, 0.01, 20)  # Linear in x
        trajectory[:, 1] = np.random.normal(0, 0.01, 20)      # Small noise in y
        trajectory[:, 2] = np.random.normal(0, 0.01, 20)      # Small noise in z
    else:
        # Create a circular trajectory
        angles = np.linspace(0, np.pi/2, 20)
        radius = 0.1
        trajectory = np.zeros((20, 3))
        trajectory[:, 0] = radius * np.cos(angles) + np.random.normal(0, 0.01, 20)
        trajectory[:, 1] = radius * np.sin(angles) + np.random.normal(0, 0.01, 20)
        trajectory[:, 2] = np.random.normal(0, 0.01, 20)
    
    return trajectory


def test_interactive_perception():
    """Test interactive perception with synthetic trajectories."""
    print("=== Testing Interactive Perception ===")
    
    # Test prismatic joint
    print("\nTesting prismatic joint detection...")
    prismatic_trajectory = create_test_trajectory('prismatic')
    ip_prismatic = InteractivePerception(prismatic_trajectory)
    params_prismatic = ip_prismatic.estimate_joint_parameters()
    
    print(f"Detected joint type: {params_prismatic['joint_type']}")
    print(f"Prismatic error: {params_prismatic['prismatic_error']:.6f}")
    print(f"Revolute error: {params_prismatic['revolute_error']:.6f}")
    
    # Test revolute joint
    print("\nTesting revolute joint detection...")
    revolute_trajectory = create_test_trajectory('revolute')
    ip_revolute = InteractivePerception(revolute_trajectory)
    params_revolute = ip_revolute.estimate_joint_parameters()
    
    print(f"Detected joint type: {params_revolute['joint_type']}")
    print(f"Prismatic error: {params_revolute['prismatic_error']:.6f}")
    print(f"Revolute error: {params_revolute['revolute_error']:.6f}")
    
    return params_prismatic, params_revolute


def test_state_constructor():
    """Test state constructor."""
    print("\n=== Testing State Constructor ===")
    
    constructor = StateConstructor()
    
    # Test prismatic state
    current_pos = np.array([0.1, 0.0, 0.0])
    initial_pos = np.array([0.0, 0.0, 0.0])
    prismatic_params = {
        'joint_type': 'prismatic',
        'joint_direction': np.array([1.0, 0.0, 0.0])
    }
    
    prismatic_state = constructor.construct_state(current_pos, initial_pos, prismatic_params)
    print(f"Prismatic state: {prismatic_state}")
    print(f"State dimension: {len(prismatic_state)}")
    
    # Test revolute state
    revolute_params = {
        'joint_type': 'revolute',
        'joint_direction': np.array([0.0, 0.0, 1.0]),
        'joint_position': np.array([0.0, 0.0, 0.0])
    }
    
    revolute_state = constructor.construct_state(current_pos, initial_pos, revolute_params)
    print(f"Revolute state: {revolute_state}")
    print(f"State dimension: {len(revolute_state)}")


def test_spot_interface():
    """Test Spot interface."""
    print("\n=== Testing Spot Interface ===")
    
    config = {'use_mock': True}
    spot = MockSpotInterface(config)
    
    # Test basic functionality
    initial_pos = spot.get_end_effector_position()
    print(f"Initial position: {initial_pos}")
    
    # Test movement
    target_pos = np.array([0.1, 0.0, 0.0])
    success = spot.move_end_effector_to(target_pos)
    print(f"Movement successful: {success}")
    
    new_pos = spot.get_end_effector_position()
    print(f"New position: {new_pos}")
    
    # Get trajectory
    trajectory = spot.get_trajectory()
    print(f"Recorded {len(trajectory)} trajectory points")
    
    spot.close()
    return spot


def test_policy_executor():
    """Test policy executor (without actual policies)."""
    print("\n=== Testing Policy Executor ===")
    
    config = {
        'checkpoint_dir': 'checkpoints',
        'action_scale': 1.0,
        'max_timesteps': 10,
        'success_threshold': 0.1
    }
    
    try:
        executor = PolicyExecutor(config)
        print("Policy executor initialized successfully")
        print(f"Loaded policies: {list(executor.policies.keys())}")
    except Exception as e:
        print(f"Policy executor initialization failed (expected without checkpoints): {e}")


def test_complete_pipeline():
    """Test the complete pipeline with mock components."""
    print("\n=== Testing Complete Pipeline ===")
    
    # Initialize components
    config = {'use_mock': True}
    spot = MockSpotInterface(config)
    
    # Set initial grasp position
    initial_pos = np.array([0.0, 0.0, 0.0])
    spot.set_initial_grasp_position(initial_pos)
    
    # Simulate wiggling and get trajectory
    wiggle_directions = [
        np.array([0.05, 0, 0]),
        np.array([-0.05, 0, 0]),
        np.array([0, 0.05, 0]),
        np.array([0, -0.05, 0])
    ]
    
    trajectory = [initial_pos]
    for direction in wiggle_directions:
        target_pos = initial_pos + direction
        spot.move_end_effector_to(target_pos)
        trajectory.append(spot.get_end_effector_position())
        spot.move_end_effector_to(initial_pos)
    
    trajectory = np.array(trajectory)
    print(f"Recorded trajectory with {len(trajectory)} points")
    
    # Interactive perception
    ip = InteractivePerception(trajectory)
    joint_params = ip.estimate_joint_parameters()
    print(f"Estimated joint type: {joint_params['joint_type']}")
    
    # State construction
    constructor = StateConstructor()
    current_pos = spot.get_end_effector_position()
    state = constructor.construct_state(current_pos, initial_pos, joint_params)
    print(f"Constructed state: {state}")
    
    # Clean up
    spot.close()
    
    print("Complete pipeline test finished")


def main():
    """Run all tests."""
    print("Spot Manipulation System - Test Suite")
    print("=" * 50)
    
    try:
        # Test individual components
        test_interactive_perception()
        test_state_constructor()
        test_spot_interface()
        test_policy_executor()
        
        # Test complete pipeline
        test_complete_pipeline()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 