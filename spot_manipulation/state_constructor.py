"""
State constructor module for building state vectors from joint parameters.
Adapted from the FLEX parallel_test.py get_state function.
"""

import numpy as np
from typing import Dict


class StateConstructor:
    """
    Constructs state vectors for policy input based on joint parameters and robot state.
    """
    
    def __init__(self):
        """Initialize the state constructor."""
        pass
    
    def construct_state(self, current_pos: np.ndarray, initial_pos: np.ndarray, 
                       joint_params: Dict) -> np.ndarray:
        """
        Construct state vector from current position, initial position, and joint parameters.
        
        Args:
            current_pos: Current end-effector position (3,)
            initial_pos: Initial grasp position (3,)
            joint_params: Dictionary containing joint parameters
            
        Returns:
            State vector for policy input
        """
        joint_type = joint_params['joint_type']
        
        if joint_type == 'prismatic':
            return self._construct_prismatic_state(current_pos, initial_pos, joint_params)
        else:
            return self._construct_revolute_state(current_pos, joint_params)
    
    def _construct_prismatic_state(self, current_pos: np.ndarray, initial_pos: np.ndarray, 
                                  joint_params: Dict) -> np.ndarray:
        """
        Construct state for prismatic joint.
        
        Args:
            current_pos: Current end-effector position
            initial_pos: Initial grasp position
            joint_params: Joint parameters dictionary
            
        Returns:
            State vector for prismatic joint
        """
        direction = joint_params['joint_direction']
        progress = current_pos - initial_pos
        
        # Concatenate direction and progress
        state = np.concatenate([direction, progress])
        return state
    
    def _construct_revolute_state(self, current_pos: np.ndarray, joint_params: Dict) -> np.ndarray:
        """
        Construct state for revolute joint.
        
        Args:
            current_pos: Current end-effector position
            joint_params: Joint parameters dictionary
            
        Returns:
            State vector for revolute joint
        """
        hinge_direction = joint_params['joint_direction']
        hinge_position = joint_params['joint_position']
        
        # Calculate projection of force point relative to hinge
        force_point = current_pos
        projection = (force_point - hinge_position - 
                     np.dot(force_point - hinge_position, hinge_direction) * hinge_direction)
        
        # Concatenate hinge direction and projection
        state = np.concatenate([hinge_direction, projection])
        return state
    
    def get_state_dimension(self, joint_type: str) -> int:
        """
        Get the dimension of the state vector for a given joint type.
        
        Args:
            joint_type: Type of joint ('prismatic' or 'revolute')
            
        Returns:
            Dimension of the state vector
        """
        if joint_type == 'prismatic':
            # direction (3) + progress (3) = 6
            return 6
        else:
            # hinge_direction (3) + projection (3) = 6
            return 6


if __name__ == '__main__':
    # Test the state constructor
    constructor = StateConstructor()
    
    # Test prismatic state
    current_pos = np.array([0.1, 0.2, 0.3])
    initial_pos = np.array([0.0, 0.0, 0.0])
    prismatic_params = {
        'joint_type': 'prismatic',
        'joint_direction': np.array([1.0, 0.0, 0.0])
    }
    
    prismatic_state = constructor.construct_state(current_pos, initial_pos, prismatic_params)
    print(f"Prismatic state: {prismatic_state}")
    print(f"Prismatic state dimension: {len(prismatic_state)}")
    
    # Test revolute state
    revolute_params = {
        'joint_type': 'revolute',
        'joint_direction': np.array([0.0, 0.0, 1.0]),
        'joint_position': np.array([0.0, 0.0, 0.0])
    }
    
    revolute_state = constructor.construct_state(current_pos, initial_pos, revolute_params)
    print(f"Revolute state: {revolute_state}")
    print(f"Revolute state dimension: {len(revolute_state)}") 