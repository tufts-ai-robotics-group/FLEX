"""
Policy executor module for loading and running trained PPO agents.
Adapted from the FLEX ppo_twohead.py implementation.
"""

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np
import os
from typing import Dict, Tuple, Optional


# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MultiheadActorCritic(nn.Module):
    """
    Multi-head actor-critic network for PPO.
    Adapted from the FLEX implementation.
    """
    
    def __init__(self, state_dim: int, action_dim: int, has_continuous_action_space: bool, 
                 action_std_init: float = 0.6):
        """
        Initialize the multi-head actor-critic network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            has_continuous_action_space: Whether action space is continuous
            action_std_init: Initial action standard deviation
        """
        super(MultiheadActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space 
        self._train = False
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        
        # Actor network
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh()
            )
            self.actor_head = {
                'strength': nn.Sequential(
                    nn.Linear(64, action_dim), 
                    nn.Tanh()
                ), 
                'direction': nn.Sequential(
                    nn.Linear(64, action_dim), 
                    nn.Tanh()
                )
            }
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh()
            )
            self.actor_head = {
                'magnitude': nn.Sequential(
                    nn.Linear(64, action_dim), 
                    nn.Softmax(dim=-1)
                ), 
                'direction': nn.Sequential(
                    nn.Linear(64, action_dim), 
                    nn.Softmax(dim=-1)
                )
            }
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64), 
            nn.Tanh(), 
            nn.Linear(64, 64), 
            nn.Tanh(), 
            nn.Linear(64, 64) 
        ) 
        self.critic_head = {
            'magnitude': nn.Sequential(
                nn.Linear(64, 1), 
                nn.Tanh()
            ), 
            'direction': nn.Sequential(
                nn.Linear(64, 1), 
                nn.Tanh()
            )
        }

    def train(self):
        """Set network to training mode."""
        self._train = True 

    def eval(self):
        """Set network to evaluation mode."""
        self._train = False

    def set_action_std(self, new_action_std: float):
        """Set action standard deviation."""
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("WARNING: Calling set_action_std() on discrete action space policy")

    def act(self, state: torch.Tensor) -> Tuple[Dict, Dict, Dict]:
        """
        Get action from current state.
        
        Args:
            state: Current state tensor
            
        Returns:
            Tuple of (action, action_logprob, state_value)
        """
        actor_feat = self.actor(state)
        critic_feat = self.critic(state)

        if self.has_continuous_action_space:
            action_mean = {
                k: self.actor_head[k](actor_feat) for k in self.actor_head.keys()
            }
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = {
                k: MultivariateNormal(action_mean[k], cov_mat) for k in action_mean.keys()
            }
        else:
            action_probs = {
                k: self.actor_head[k](actor_feat) for k in self.actor_head.keys()
            }
            dist = {
                k: Categorical(action_probs[k]) for k in action_probs.keys()
            }
        
        if self._train:
            action = {
                k: dist[k].sample().detach() for k in dist.keys()
            } 
        else:
            action = {
                k: dist[k].mode for k in dist.keys()
            }
        
        action_logprob = {
            k: dist[k].log_prob(v).detach() for k, v in action.items()
        }
        state_val = {
            k: self.critic_head[k](critic_feat).detach() for k in self.critic_head.keys()
        }

        return action, action_logprob, state_val


class PolicyExecutor:
    """
    Executes trained PPO policies for robot manipulation.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the policy executor.
        
        Args:
            config: Configuration dictionary containing policy paths and parameters
        """
        self.config = config
        self.policies = {}
        self.action_scale = config.get('action_scale', 1.0)
        self.max_timesteps = config.get('max_timesteps', 1000)
        self.success_threshold = config.get('success_threshold', 0.1)
        
        # Load policies
        self._load_policies()
    
    def _load_policies(self):
        """Load trained policies for both joint types."""
        checkpoint_dir = self.config['checkpoint_dir']
        
        # Load prismatic policy
        prismatic_path = os.path.join(checkpoint_dir, 'prismatic_final.pth')
        if os.path.exists(prismatic_path):
            self.policies['prismatic'] = self._load_policy(prismatic_path)
            print(f"Loaded prismatic policy from {prismatic_path}")
        else:
            print(f"Warning: Prismatic policy not found at {prismatic_path}")
        
        # Load revolute policy
        revolute_path = os.path.join(checkpoint_dir, 'revolute_final.pth')
        if os.path.exists(revolute_path):
            self.policies['revolute'] = self._load_policy(revolute_path)
            print(f"Loaded revolute policy from {revolute_path}")
        else:
            print(f"Warning: Revolute policy not found at {revolute_path}")
    
    def _load_policy(self, checkpoint_path: str) -> MultiheadActorCritic:
        """
        Load a policy from checkpoint.
        
        Args:
            checkpoint_path: Path to policy checkpoint
            
        Returns:
            Loaded policy network
        """
        # Create policy network (assuming 6D state, 3D action)
        policy = MultiheadActorCritic(
            state_dim=6,  # 6D state as per state constructor
            action_dim=3,  # 3D action (dx, dy, dz)
            has_continuous_action_space=True,
            action_std_init=0.6
        ).to(device)
        
        # Load weights
        policy.load_state_dict(torch.load(checkpoint_path, map_location=device))
        policy.eval()
        
        return policy
    
    def execute_policy(self, joint_type: str, initial_state: np.ndarray, 
                      spot_interface) -> bool:
        """
        Execute policy for manipulation.
        
        Args:
            joint_type: Type of joint ('prismatic' or 'revolute')
            initial_state: Initial state vector
            spot_interface: Interface to Spot robot
            
        Returns:
            bool: True if manipulation was successful
        """
        if joint_type not in self.policies:
            print(f"Error: Policy for joint type '{joint_type}' not loaded")
            return False
        
        policy = self.policies[joint_type]
        current_state = initial_state.copy()
        last_grasp_pos = spot_interface.get_end_effector_position()
        
        print(f"Executing {joint_type} policy...")
        
        for timestep in range(self.max_timesteps):
            # Get action from policy
            action = self._get_action(policy, current_state)
            
            # Apply action to robot
            success = self._apply_action(action, last_grasp_pos, spot_interface)
            if not success:
                print("Failed to apply action to robot")
                return False
            
            # Update state
            new_pos = spot_interface.get_end_effector_position()
            current_state = self._update_state(current_state, new_pos, joint_type)
            last_grasp_pos = new_pos
            
            # Check for success
            if self._check_success(new_pos, spot_interface):
                print(f"Manipulation successful after {timestep + 1} timesteps")
                return True
            
            # Optional: Add delay for real robot
            import time
            time.sleep(0.1)
        
        print(f"Manipulation failed after {self.max_timesteps} timesteps")
        return False
    
    def _get_action(self, policy: MultiheadActorCritic, state: np.ndarray) -> np.ndarray:
        """
        Get action from policy.
        
        Args:
            policy: Policy network
            state: Current state
            
        Returns:
            Action vector
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_dict, _, _ = policy.act(state_tensor)
            
            # Combine strength and direction for continuous action
            if 'strength' in action_dict and 'direction' in action_dict:
                strength = action_dict['strength'].cpu().numpy().flatten()
                direction = action_dict['direction'].cpu().numpy().flatten()
                action = strength * direction
            else:
                # Fallback for discrete action or single action
                action = list(action_dict.values())[0].cpu().numpy().flatten()
            
            # Apply action scaling
            action = action * self.action_scale
            
            return action
    
    def _apply_action(self, action: np.ndarray, last_grasp_pos: np.ndarray, 
                     spot_interface) -> bool:
        """
        Apply action to robot.
        
        Args:
            action: Action vector (dx, dy, dz)
            last_grasp_pos: Last grasp position
            spot_interface: Interface to Spot robot
            
        Returns:
            bool: True if action was applied successfully
        """
        try:
            # Calculate new position
            new_pos = last_grasp_pos + action
            
            # Move robot to new position
            success = spot_interface.move_end_effector_to(new_pos)
            return success
        except Exception as e:
            print(f"Error applying action: {e}")
            return False
    
    def _update_state(self, current_state: np.ndarray, new_pos: np.ndarray, 
                     joint_type: str) -> np.ndarray:
        """
        Update state based on new position.
        
        Args:
            current_state: Current state vector
            new_pos: New end-effector position
            joint_type: Type of joint
            
        Returns:
            Updated state vector
        """
        # For now, we'll need to reconstruct the state with the new position
        # This is a simplified version - in practice, you'd need the full joint parameters
        # and initial position to properly reconstruct the state
        
        # For prismatic: [direction(3), progress(3)]
        if joint_type == 'prismatic':
            # Keep direction, update progress
            direction = current_state[:3]
            initial_pos = np.array([0, 0, 0])  # This should come from the system
            progress = new_pos - initial_pos
            return np.concatenate([direction, progress])
        
        # For revolute: [hinge_direction(3), projection(3)]
        else:
            # Keep hinge direction, update projection
            hinge_direction = current_state[:3]
            hinge_position = np.array([0, 0, 0])  # This should come from joint params
            projection = (new_pos - hinge_position - 
                         np.dot(new_pos - hinge_position, hinge_direction) * hinge_direction)
            return np.concatenate([hinge_direction, projection])
    
    def _check_success(self, current_pos: np.ndarray, spot_interface) -> bool:
        """
        Check if manipulation task is successful.
        
        Args:
            current_pos: Current end-effector position
            spot_interface: Interface to Spot robot
            
        Returns:
            bool: True if task is successful
        """
        # This is a placeholder - implement based on your success criteria
        # For example, check if object is in desired position or orientation
        
        # Simple example: check if moved far enough from initial position
        initial_pos = spot_interface.get_initial_grasp_position()
        distance = np.linalg.norm(current_pos - initial_pos)
        
        return distance > self.success_threshold


if __name__ == '__main__':
    # Test the policy executor
    config = {
        'checkpoint_dir': 'checkpoints',
        'action_scale': 1.0,
        'max_timesteps': 100,
        'success_threshold': 0.1
    }
    
    executor = PolicyExecutor(config)
    print("Policy executor initialized") 