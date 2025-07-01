import os
import numpy as np


class PolicyManager:
    """
    Manages TD3 policy loading and execution for different joint types.
    Handles the manipulation task by converting policy actions to robot commands.
    """
    
    def __init__(self, models_dir="models"):
        """
        Args:
            models_dir: Directory containing policy models
        """
        self.models_dir = models_dir
        self.current_policy = None
        self.current_joint_type = None
        
    def load_policy(self, joint_type):
        """
        Load TD3 policy for specified joint type.
        
        Args:
            joint_type: "prismatic" or "revolute"
            
        Returns:
            TD3 policy object
        """
        from td3 import TD3
        
        # Policy parameters (match your trained models)
        state_dim = 6
        action_dim = 3  
        max_action = 1.0
        
        # Load appropriate model
        model_path = os.path.join(self.models_dir, joint_type, "best_model")
        
        if not os.path.exists(f"{model_path}_actor.pth"):
            raise FileNotFoundError(f"Policy not found: {model_path}_actor.pth")
        
        # Create and load policy
        policy = TD3(0.001, state_dim, action_dim, max_action)
        policy.load_actor(self.models_dir + f"/{joint_type}", "best_model")
        
        self.current_policy = policy
        self.current_joint_type = joint_type
        print(f"→ Loaded {joint_type} policy")
        
        return policy
    
    def get_policy_action(self, state):
        """
        Get action from current policy given state.
        
        Args:
            state: State vector from interactive perception
            
        Returns:
            np.array: Action vector [force_x, force_y, force_z]
        """
        if self.current_policy is None:
            raise ValueError("No policy loaded!")
        
        # Get action from policy
        action = self.current_policy.select_action(state)
        
        return action
    
    def convert_action_to_target_position(self, action, current_position, action_scale=0.02):
        """
        Convert policy action to target position for robot.
        
        Args:
            action: Policy action [force_x, force_y, force_z] 
            current_position: Current gripper position [x, y, z]
            action_scale: Scale factor for action (meters)
            
        Returns:
            np.array: Target position [x, y, z]
        """
        # Scale action and add to current position
        scaled_action = action * action_scale
        target_position = current_position + scaled_action
        
        return target_position
    
    def execute_policy_step(self, state, current_position, robot_interface, action_scale=0.02):
        """
        Execute one policy step: get action, convert to command, send to robot.
        
        Args:
            state: Current state vector
            current_position: Current gripper position
            robot_interface: Robot interface for sending commands
            action_scale: Scale factor for actions
            
        Returns:
            tuple: (action, target_position)
        """
        # Get action from policy
        action = self.get_policy_action(state)
        
        # Convert to target position
        target_position = self.convert_action_to_target_position(
            action, current_position, action_scale
        )
        
        # Send command to robot
        robot_interface.move_arm_to_position(target_position)
        
        return action, target_position