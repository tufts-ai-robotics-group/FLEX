'''
General training script for the experiments
'''

import numpy as np

import robosuite as suite
import yaml
import json

from agent.st3_td3 import CustomTD3Policy
from env.train_multiple_revolute_env import MultipleRevoluteEnv 
from env.train_prismatic_env import TrainPrismaticEnv 
from env.object_env import GeneralObjectEnv
from env.wrappers import ActionRepeatWrapperNew, make_env, SubprocessEnv
import time
import random
import os
import stable_baselines3 as sb3 
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback


class SuccessProjectionCallBack(BaseCallback):
    def __init__(self, n_envs=24, log_interval=100000, verbose=0):
        super(SuccessProjectionCallBack, self).__init__(verbose)
        self.n_success = []
        self.n_envs = n_envs
        self.total_episodes = 0
        self.log_interval = log_interval
        self.timesteps_since_last_log = 0
        self.n_timesteps = 0
        self.projection_ratios = []
        self.episode_projections = [[] for i in range(n_envs)]
        self.projections = []
    
    def _on_step(self):
        self.n_timesteps += 1
        self.timesteps_since_last_log += 1
        # print(self.training_env)
        envs = self.training_env
        # print(len(envs.get_attr('current_step_force_projection')))
        projections = envs.get_attr('current_step_force_projection') 
        successes = envs.get_attr('success')
        dones = envs.get_attr('done') 
        print(dones)
        for i in range(self.n_envs):
            self.episode_projections[i].append(projections[i])
            if dones[i]:
                self.projections.append(np.mean(self.episode_projections[i]))
                self.episode_projections[i] = []
                self.total_episodes += 1
                self.n_success.append(successes[i])
        # for i, env in enumerate(envs):
        #     print(env.send(('get_attr', 'current_step_force_projection')))
        #     self.episode_projections[i].append(env.get_attr('current_step_force_projection'))
        #     if env.done:
        #         self.projections.append(np.mean(self.episode_projections[i]))
        #         self.episode_projections[i] = []
        #         self.total_episodes += 1
        #         self.n_success.append(self._check_success()) 

        if self.timesteps_since_last_log >= self.log_interval:
            self._log_success_rates_and_projections()
            self.timesteps_since_last_log = 0
        return True 
    
    def _log_success_rates_and_projections(self):
        success_rate = np.mean(self.n_success[-50])
        print(f'Current success rate (last 50 episodes): {success_rate}')
        print(f'Current average projection (last 50 episodes): {np.mean(self.projections)}')
        return success_rate
    
    def _early_stopping(self):
        if self.total_episodes > 400:
            averate_projection = np.mean(self.projections[-100:])
            if averate_projection >= 0.75:
                print(f'Projection ratio reached 0.75 after {self.n_timesteps} timesteps, {self.total_episodes} episodes')
                print(f'Current success rate: {np.mean(self.n_success[-50])}')
                return True
        return False


revolute_state = lambda state: np.concatenate([
    state['hinge_direction'], 
    state['force_point']-state['hinge_position']-np.dot(state['force_point']-state['hinge_position'], state['hinge_direction'])*state['hinge_direction']
    ]) 

prismatic_state = lambda obs: np.concatenate([obs["joint_direction"], obs["force_point_relative_to_start"]])

def train(run_id):
    with open('cfg/train_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Experiment Settings
    experiment_name = config['experiment_name']
    train_objects = config['train_objects'] 
    test_objects = config['test_objects'] 
    save_policy_freq = config['save_policy_freq'] 
    n_envs = config['n_envs']

    # Force RL agent hyperparameters
    rl_config = config['force_policy'] 
    gamma = rl_config['gamma']
    if rl_config['algorithm'] == 'ddpg':
        lr = rl_config['lr']
        batch_size = rl_config['batch_size']
        polyak = rl_config['polyak']              
        noise_clip = 1
        policy_delay = 2            
    else:
        raise Exception(f'{rl_config["algorithm"]} is not implemented')

    # Environment settings
    env_config = config['environment']
    max_timesteps = env_config['max_timesteps']
    action_repeat = env_config['action_repeat'] 

    pris_force_policy_dir = f'checkpoints/force_policies/{experiment_name}/prismatic' 
    rev_force_policy_dir = f'checkpoints/force_policies/{experiment_name}/revolute'
    os.makedirs(pris_force_policy_dir, exist_ok=True)
    os.makedirs(rev_force_policy_dir, exist_ok=True) 

    print(f'prismatic policy is saving to : {pris_force_policy_dir}')
    print(f'revolute policy is saving to : {rev_force_policy_dir}')

    n_runs_pris = sum(os.path.isdir(os.path.join(pris_force_policy_dir, entry)) for entry in os.listdir(pris_force_policy_dir))
    # assert n_runs == sum(os.path.isdir(os.path.join(rev_force_policy_dir, entry)) for entry in os.listdir(rev_force_policy_dir))
    n_runs_rev = sum(os.path.isdir(os.path.join(rev_force_policy_dir, entry)) for entry in os.listdir(rev_force_policy_dir))

    pris_force_policy_dir += '/' + str(n_runs_pris)
    rev_force_policy_dir += '/' + str(n_runs_rev)

    os.makedirs(pris_force_policy_dir, exist_ok=True)
    os.makedirs(rev_force_policy_dir, exist_ok=True) 

    log_dir = f'logs/{experiment_name}' 
    os.makedirs(log_dir, exist_ok=True)
    n_runs = sum(os.path.isdir(os.path.join(log_dir, entry)) for entry in os.listdir(log_dir)) 
    log_dir += '/' + str(n_runs) 
    os.makedirs(log_dir, exist_ok=True)

    rollouts = 5
    state_dim = 6
    action_dim = 3
    max_action = float(5)
    # prismatic_policy = TD3.load(prismatic_policy, directory=pris_force_policy_dir)

    policy_noise_schedule = [
        1, 
        0.4, 
        0.2, 
        0.1
    ]
    # Curriculum for training 
    # Each curriculum is a tuple of (object pose range, use random point, object type)
    curriculas = [
        #  ((-np.pi, np.pi), True, ['train-drawer-1']),
         ((-np.pi / 2.0, -np.pi / 2), True, ['train-microwave-1', 'train-dishwasher-1']),
         ((-np.pi / 2.0, -np.pi / 2.0), True, ['train-microwave-1', 'train-dishwasher-1']),
         ((-np.pi, 0), True, ['train-microwave-1', 'train-dishwasher-1']),
         ((-np.pi, 0), True, ['train-microwave-1', 'train-dishwasher-1']),
         ((-np.pi, np.pi), True, ['train-microwave-1', 'train-dishwasher-1']),
         ((-np.pi, np.pi), True, ['train-microwave-1', 'train-dishwasher-1']),
        #  ((-np.pi, np.pi), True, "dishwasher-like"),
        #  ((-np.pi, np.pi), True, "all"),
    ]

    episodes_schedule = [
        # 2000,
        1000,
        1000, 
        2000, 
        2000, 
        2000,
        3000, 
    ]

    assert len(curriculas) == len(episodes_schedule), "The length of curriculas and episodes_schedule should be the same." 
    curriculum_env_kwargs = {
            "init_object_angle": (-np.pi/2, 0),
            "has_renderer": False,
            "has_offscreen_renderer": False,
            "use_camera_obs": False,
            "control_freq": 20,
            "horizon": max_timesteps,
            "reward_scale": 1.0,
            "random_force_point": True,
            "object_name": 'train-drawer-1', 
            "env_name": 'TrainPrismaticEnv'
        } 

    # env_kwarg_list = [
    #     curriculum_env_kwargs for i in range(n_envs)
    # ]
    # for i in range(n_envs):
    #     env_kwarg_list[i]['object_name'] = random.choice(cur_curriculum_objects) if not isinstance(cur_curriculum_objects, str) else cur_curriculum_objects 
    #     print(f'process {i} is training with object {env_kwarg_list[i]["object_name"]}')
    env = SubprocVecEnv([make_env(curriculum_env_kwargs) for i in range(n_envs)])

    time_spent_on_training = 0. 

    prismatic_time_spent = 0.
    action_noise = sb3.common.noise.NormalActionNoise(mean=np.zeros(action_dim), sigma=0.1 * np.ones(action_dim))
    callback = SuccessProjectionCallBack(log_interval=10000)

    prismatic_policy = sb3.TD3(
        CustomTD3Policy, 
        env, 
        action_noise=action_noise, 
        verbose=1
    ) 
    prismatic_policy.learn(total_timesteps=2000*200, callback=callback)

    revolute_policy = sb3.TD3(
        CustomTD3Policy, 
        env, 
        action_noise=action_noise, 
        verbose=1
    )

    for curriculum_idx, current_curriculum in enumerate(curriculas):
        # some list to record training data
        
        cur_curriculum_objects = current_curriculum[-1]
            
        current_episode_object = random.choice(cur_curriculum_objects) if not isinstance(cur_curriculum_objects, str) else cur_curriculum_objects
        # print(f'episode {episodes} is training with object {current_episode_object}')

        curriculum_env_kwargs = {
            "init_object_angle": current_curriculum[0],
            "has_renderer": False,
            "has_offscreen_renderer": False,
            "use_camera_obs": False,
            "control_freq": 20,
            "horizon": max_timesteps,
            "reward_scale": 1.0,
            "random_force_point": current_curriculum[1],
            "object_name": current_episode_object
        } 
        curriculum_env_kwargs['env_name'] = 'MultipleRevoluteEnv' 

        env_kwarg_list = [
            curriculum_env_kwargs for i in range(n_envs)
        ]
        for i in range(n_envs):
            env_kwarg_list[i]['object_name'] = random.choice(cur_curriculum_objects) if not isinstance(cur_curriculum_objects, str) else cur_curriculum_objects 
            print(f'process {i} is training with object {env_kwarg_list[i]["object_name"]}')
        env = SubprocVecEnv([make_env(env_kwarg_list[i]) for i in range(n_envs)])
        action_noise = sb3.common.noise.NormalActionNoise(mean=np.zeros(action_dim), sigma=0.1 * np.ones(action_dim))
        revolute_policy.set_env(env)
        revolute_policy.learn(total_timesteps=episodes_schedule[curriculum_idx]*200, callback=callback)




if __name__ == "__main__":

    # file_dir = os.path.dirname(os.path.abspath(__file__))
    # project_dir = os.path.dirname(file_dir)
    # output_dir = os.path.join(file_dir, "force_training_outputs")
    # checkpoint_dir = os.path.join(project_dir, "checkpoints/force_policies")

    # # create the output directory if it does not exist
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    # # create the checkpoint directory if it does not exist
    # if not os.path.exists(checkpoint_dir):
    #     os.makedirs(checkpoint_dir)

    # algo_name = "curriculum_door_continuous_random_point_td3"
    # for trial in range(10):
    #     train(trial, output_dir, algo_name, checkpoint_dir)
    train(0)
    
