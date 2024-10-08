import numpy as np

import robosuite as suite
import yaml
import json

from agent.td3 import TD3, ReplayBuffer
from env.train_multiple_revolute_env import MultipleRevoluteEnv
from env.wrappers import ActionRepeatWrapperNew
import time
import os


gamma = 0.99                # discount for future rewards
batch_size = 100            # num of transitions sampled from replay buffer
lr = 0.001
polyak = 0.995              # target policy update parameter (1-tau)
noise_clip = 1
policy_delay = 2            # delayed policy updates parameter
max_timesteps = 200       # max timesteps in one episode
rollouts = 5
action_repeat = 1 
obs_dim = 6
action_dim = 3
max_action = float(5)

policy_dir = 'checkpoints/force_policies'
policy_name = 'curriculum_door_continuous_random_point_td3_0_curriculum_8'
policy = TD3(lr, obs_dim, action_dim, max_action)
policy.load(policy_dir, policy_name)



curriculum_env_kwargs = {
                "init_door_angle": (-np.pi/2, -np.pi/2),
                "has_renderer": True,
                "has_offscreen_renderer": True,
                "use_camera_obs": False,
                "control_freq": 20,
                "horizon": 1000,
                "reward_scale": 1.0,
                "random_force_point": True,
                "object_name": "train-dishwasher-1"
            }

env: MultipleRevoluteEnv = suite.make(
                "MultipleRevoluteEnv", 
                **curriculum_env_kwargs
                )

obs = env.reset()
h_point, f_point, h_direction = obs['hinge_position'], obs['force_point'], obs['hinge_direction']
obs = np.concatenate([
                        h_direction, 
                        f_point-h_point-np.dot(f_point-h_point, h_direction)*h_direction
                        ])


rollouts = 10

for i in range(rollouts):
    obs = env.reset()
    h_point, f_point, h_direction = obs['hinge_position'], obs['force_point'], obs['hinge_direction']
    obs = np.concatenate([
                        h_direction, 
                        f_point-h_point-np.dot(f_point-h_point, h_direction)*h_direction
                        ])
    done = False
    for t in range(max_timesteps):
        action = policy.select_action(obs)
        print(action)
        obs, reward, done, _ = env.step(action)
        env.render()
        h_point, f_point, h_direction = obs['hinge_position'], obs['force_point'], obs['hinge_direction']
        obs = np.concatenate([
                        h_direction, 
                        f_point-h_point-np.dot(f_point-h_point, h_direction)*h_direction
                        ])
        if done:
            break


