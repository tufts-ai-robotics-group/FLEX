#!/usr/bin/env python3
"""
Main entry point for Spot robot manipulation system.
Implements the complete pipeline: interactive perception -> state estimation -> policy execution.
"""

import numpy as np
import argparse
import time
import json
import os
from typing import Dict, Tuple, List

from interactive_perception import InteractivePerception
from spot_interface import SpotInterface
from policy_executor import PolicyExecutor
from state_constructor import StateConstructor


class SpotManipulationSystem:
    """
    Main system that orchestrates the complete manipulation pipeline.
    """
    
    def __init__(self, config_path: str = "config/spot_config.json"):
        """
        Initialize the manipulation system.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.spot_interface = SpotInterface(self.config['spot'])
        self.state_constructor = StateConstructor()
        self.policy_executor = PolicyExecutor(self.config['policy'])
        
        # Interactive perception will be created when needed
        self.interactive_perception = None
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def run_manipulation_pipeline(self, target_point: np.ndarray) -> bool:
        """
        Execute the complete manipulation pipeline.
        
        Args:
            target_point: 3D point in world coordinates to manipulate
            
        Returns:
            bool: True if manipulation was successful
        """
        print("=== Starting Spot Manipulation Pipeline ===")
        
        try:
            # Step 1: Grasp the object (assumed already done)
            print("Step 1: Object already grasped")
            
            # Step 2: Interactive perception - wiggle and estimate joint parameters
            print("Step 2: Performing interactive perception...")
            joint_params = self._perform_interactive_perception()
            
            if joint_params is None:
                print("Failed to estimate joint parameters")
                return False
            
            print(f"Estimated joint type: {joint_params['joint_type']}")
            print(f"Joint direction: {joint_params['joint_direction']}")
            
            # Step 3: Construct state from estimated parameters
            print("Step 3: Constructing state...")
            state = self._construct_state(joint_params)
            
            # Step 4: Execute policy
            print("Step 4: Executing manipulation policy...")
            success = self._execute_policy(joint_params['joint_type'], state)
            
            print(f"Manipulation {'successful' if success else 'failed'}")
            return success
            
        except Exception as e:
            print(f"Error in manipulation pipeline: {e}")
            return False
    
    def _perform_interactive_perception(self) -> Dict:
        """
        Perform interactive perception by wiggling the object.
        
        Returns:
            Dict containing estimated joint parameters
        """
        # Define wiggle directions: forward, backward, left, right
        wiggle_directions = [
            np.array([0.05, 0, 0]),   # Forward
            np.array([-0.05, 0, 0]),  # Backward
            np.array([0, 0.05, 0]),   # Left
            np.array([0, -0.05, 0]),  # Right
        ]
        
        trajectory = []
        
        # Get initial position
        initial_pos = self.spot_interface.get_end_effector_position()
        trajectory.append(initial_pos)
        
        # Perform wiggling in each direction
        for direction in wiggle_directions:
            print(f"Wiggling in direction: {direction}")
            
            # Move to wiggled position
            target_pos = initial_pos + direction
            self.spot_interface.move_end_effector_to(target_pos)
            time.sleep(0.5)  # Wait for movement to complete
            
            # Record position
            current_pos = self.spot_interface.get_end_effector_position()
            trajectory.append(current_pos)
            
            # Return to initial position
            self.spot_interface.move_end_effector_to(initial_pos)
            time.sleep(0.5)
        
        # Convert trajectory to numpy array
        trajectory = np.array(trajectory)
        
        # Use interactive perception to estimate joint parameters
        self.interactive_perception = InteractivePerception(trajectory)
        
        # Compare prismatic vs revolute errors
        prismatic_error, prismatic_direction = self.interactive_perception.prismatic_error()
        revolute_error, revolute_center, revolute_radius, revolute_axis = self.interactive_perception.revolute_error()
        
        print(f"Prismatic error: {prismatic_error}")
        print(f"Revolute error: {revolute_error}")
        
        # Determine joint type based on error comparison
        if prismatic_error < revolute_error:
            joint_type = 'prismatic'
            joint_direction = prismatic_direction
            joint_position = None
        else:
            joint_type = 'revolute'
            joint_direction = revolute_axis
            joint_position = revolute_center
        
        return {
            'joint_type': joint_type,
            'joint_direction': joint_direction,
            'joint_position': joint_position,
            'prismatic_error': prismatic_error,
            'revolute_error': revolute_error
        }
    
    def _construct_state(self, joint_params: Dict) -> np.ndarray:
        """
        Construct state vector from joint parameters.
        
        Args:
            joint_params: Estimated joint parameters
            
        Returns:
            State vector for policy input
        """
        current_pos = self.spot_interface.get_end_effector_position()
        initial_pos = self.spot_interface.get_initial_grasp_position()
        
        return self.state_constructor.construct_state(
            current_pos=current_pos,
            initial_pos=initial_pos,
            joint_params=joint_params
        )
    
    def _execute_policy(self, joint_type: str, initial_state: np.ndarray) -> bool:
        """
        Execute the manipulation policy.
        
        Args:
            joint_type: Type of joint ('prismatic' or 'revolute')
            initial_state: Initial state vector
            
        Returns:
            bool: True if manipulation was successful
        """
        return self.policy_executor.execute_policy(
            joint_type=joint_type,
            initial_state=initial_state,
            spot_interface=self.spot_interface
        )


def main():
    """Main function to run the manipulation system."""
    parser = argparse.ArgumentParser(description="Spot Robot Manipulation System")
    parser.add_argument("--config", default="config/spot_config.json", 
                       help="Path to configuration file")
    parser.add_argument("--target_x", type=float, default=0.0,
                       help="Target X coordinate")
    parser.add_argument("--target_y", type=float, default=0.0,
                       help="Target Y coordinate")
    parser.add_argument("--target_z", type=float, default=0.0,
                       help="Target Z coordinate")
    
    args = parser.parse_args()
    
    # Create target point
    target_point = np.array([args.target_x, args.target_y, args.target_z])
    
    # Initialize and run system
    system = SpotManipulationSystem(args.config)
    success = system.run_manipulation_pipeline(target_point)
    
    if success:
        print("Manipulation completed successfully!")
    else:
        print("Manipulation failed.")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main()) 