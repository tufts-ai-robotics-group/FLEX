'''
General training script for the experiments
'''

import numpy as np

import robosuite as suite
import yaml
import json

# from agent.td3 import TD3, ReplayBuffer
from agent.deep_td3 import TD3, ReplayBuffer
from env.train_multiple_revolute_env import MultipleRevoluteEnv 
from env.train_prismatic_env import TrainPrismaticEnv 
from env.object_env import GeneralObjectEnv
from env.wrappers import ActionRepeatWrapperNew, make_env, SubprocessEnv
import time
import random
import os
import stable_baselines3 as sb3 
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


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
        noise_clip = 2
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


    n_runs_pris = sum(os.path.isdir(os.path.join(pris_force_policy_dir, entry)) for entry in os.listdir(pris_force_policy_dir)) + 1
    # assert n_runs == sum(os.path.isdir(os.path.join(rev_force_policy_dir, entry)) for entry in os.listdir(rev_force_policy_dir))
    n_runs_rev = sum(os.path.isdir(os.path.join(rev_force_policy_dir, entry)) for entry in os.listdir(rev_force_policy_dir)) + 1

    pris_force_policy_dir += '/' + str(n_runs_pris)
    rev_force_policy_dir += '/' + str(n_runs_rev)

    os.makedirs(pris_force_policy_dir, exist_ok=True)
    os.makedirs(rev_force_policy_dir, exist_ok=True) 

    print(f'prismatic policy is saving to : {pris_force_policy_dir}')
    print(f'revolute policy is saving to : {rev_force_policy_dir}')

    log_dir = f'logs/{experiment_name}' 
    os.makedirs(log_dir, exist_ok=True)
    n_runs = sum(os.path.isdir(os.path.join(log_dir, entry)) for entry in os.listdir(log_dir)) 
    log_dir += '/' + str(n_runs) 
    os.makedirs(log_dir, exist_ok=True)

    rollouts = 5
    state_dim = 6
    action_dim = 3
    max_action = float(5)

    cur_run_rewards = []
    cur_run_projections = [] 

    cur_run_eval_rwds = [] 
    cur_run_eval_projections = [] 
    cur_run_eval_success_rates = []


    prismatic_policy = TD3(lr, state_dim, action_dim, max_action)
    revolute_policy = TD3(lr, state_dim, action_dim, max_action) 
    # prismatic_polic
    prismatic_replay_buffer = ReplayBuffer() 
    revolute_replay_buffer = ReplayBuffer() 

    # prismatic_policy = TD3.load(prismatic_policy, directory=pris_force_policy_dir)

    policy_noise_schedule = [
        1.6,
        1.2,
        1, 
        0.8, 
        0.6,
        0.4, 
        0.2, 
        0.1
    ]
    # Curriculum for training 
    # Each curriculum is a tuple of (object pose range, use random point, object type)
    curriculas = [
        #  ((-np.pi, np.pi), True, ['train-drawer-1']),
         ((-np.pi / 2.0, -np.pi / 2), False, ['train-microwave-1', 'train-dishwasher-1']),
         ((-np.pi / 2.0, -np.pi / 2.0), True, ['train-microwave-1', 'train-dishwasher-1']),
         ((-np.pi/2, 0), False, ['train-microwave-1', 'train-dishwasher-1']),
         ((-np.pi/2, 0), True, ['train-microwave-1', 'train-dishwasher-1']),
         ((-np.pi, 0), False, ['train-microwave-1', 'train-dishwasher-1']),
         ((-np.pi, 0), True, ['train-microwave-1', 'train-dishwasher-1']),
         ((-np.pi, np.pi), False, ['train-microwave-1', 'train-dishwasher-1']),
         ((-np.pi, np.pi), True, ['train-microwave-1', 'train-dishwasher-1']),
        #  ((-np.pi/2, 0), True, ['train-microwave-1', 'train-dishwasher-1']),
        #  ((-np.pi, np.pi), True, ['train-microwave-1', 'train-dishwasher-1']),
        #  ((-np.pi, np.pi), True, "dishwasher-like"),
        #  ((-np.pi, np.pi), True, "all"),
    ]

    episodes_schedule = [
        # 300,
        # 300,
        2000, 
        2000, 
        4000, 
        4000,
        6000,
        6000,
        10000,
        10000, 
    ]

    assert len(curriculas) == len(episodes_schedule), "The length of curriculas and episodes_schedule should be the same." 

    # Prismatic training and saving!
    prismatic_noise_idx = 0
    prismatic_training_timesteps = 0
    prismatic_training_time = 0.
    per_ep_projections = [[] for i in range(n_envs)]
    per_ep_rewards = [0 for i in range(n_envs)]
    prismatic_rewards = []
    prismatic_projections = []
    prismatic_success = []
    prismatic_env_kwargs = {
            "init_object_angle": (-np.pi/2, -np.pi/2),
            "has_renderer": False,
            "has_offscreen_renderer": False,
            "use_camera_obs": False,
            "control_freq": 20,
            "horizon": max_timesteps,
            "reward_scale": 1.0,
            "random_force_point": True,
            "object_name": 'train-drawer-1'
    } 

    prismatic_curricula = [
        ((-np.pi/2, -np.pi/2), False) ,
        ((-np.pi/2, -np.pi/2), True), 
        ((-np.pi/2, 0), False),
        ((-np.pi/2, 0), True) ,
        ((-np.pi, 0), False),
        ((-np.pi, 0), True), 
        ((-np.pi, np.pi), False),
        ((-np.pi, np.pi), True)
    ]
    prismatic_episodes = [
        2000, 
        2000, 
        5000,
        10000,
        20000, 
        20000, 
        30000,
        30000
    ]

    for c_idx, curriculum in enumerate(prismatic_curricula):
        cur_episodes = prismatic_episodes[c_idx]
        prismatic_env_kwargs['env_name'] = 'TrainPrismaticEnv'
        prismatic_env_kwargs['init_object_angle'] = curriculum[0]
        prismatic_env_kwargs['random_force_point'] = curriculum[1]
        prismatic_env = SubprocVecEnv([make_env(prismatic_env_kwargs) for i in range(n_envs)])
        policy = prismatic_policy
        replay_buffer = prismatic_replay_buffer
        prismatic_training_start = time.time()
        state  = prismatic_env.reset()
        while prismatic_training_timesteps < cur_episodes * 200:
            action = policy.select_action(state)
            successes = prismatic_env.get_attr('success')
            next_state, reward, done, _ = prismatic_env.step(action)
            projections = prismatic_env.get_attr('current_step_force_projection')
            for i in range(n_envs):
                replay_buffer.add((state[i], action[i], reward[i], next_state[i], float(done[i])))
                per_ep_projections[i].append(projections[i])
                per_ep_rewards[i] += reward[i]
                if done[i]:
                    prismatic_projections.append(np.mean(per_ep_projections[i]))
                    per_ep_projections[i] = []
                    prismatic_success.append(successes[i])
                    prismatic_rewards.append(per_ep_rewards[i])
                    per_ep_rewards[i] = 0
            prismatic_training_timesteps += len(state)
            if prismatic_training_timesteps % 100*200 == 0:
                prismatic_noise_idx  = min(prismatic_noise_idx + 1, len(policy_noise_schedule)-1)
            if prismatic_training_timesteps % 48 == 0:
                policy.update(replay_buffer, 48, batch_size, gamma, polyak, policy_noise_schedule[prismatic_noise_idx], noise_clip, policy_delay)
            if prismatic_training_timesteps % 10000*n_envs == 0:
                policy.save(pris_force_policy_dir, f'step_{prismatic_training_timesteps}')
            if prismatic_training_timesteps % 10000 == 0:
                print(f'Prismatic training timesteps: {prismatic_training_timesteps}')
                print(f'Prismatic projection ratio: {np.mean(prismatic_projections[-100:])}')
                print(f'Prismatic reward: {np.mean(prismatic_rewards[-100:])}')
                print(f'Prismatic success rate: {np.mean(prismatic_success[-100:])}')
            if prismatic_training_timesteps > 0.1 * cur_episodes * 200 and np.mean(prismatic_projections[-50]) > 0.95:
                print(f'Prismatic reached the projection threshold in {prismatic_training_timesteps} steps')
                break
        prismatic_training_end = time.time()
        prismatic_training_time += prismatic_training_end - prismatic_training_start
        policy.save(pris_force_policy_dir, f'prismatic_{c_idx}_final')
        prismatic_env.close()
    policy.save(pris_force_policy_dir, f'prismatic_final')
    json_dict = {
        'timesteps': prismatic_training_timesteps,
        'time_spent': prismatic_training_time
    }
    
    with open(pris_force_policy_dir + '/training_time.json', 'w') as f:
        json.dump(json_dict, f)

    # Revolute training and saving!
    time_spent_on_training = 0. 
    current_training_timesteps = 0
    policy = revolute_policy
    replay_buffer = revolute_replay_buffer
    for curriculum_idx, current_curriculum in enumerate(curriculas):
        cur_noise_idx = 0
        cur_curriculum_objects = current_curriculum[-1]         
        curriculum_env_kwargs = {
            "init_object_angle": current_curriculum[0],
            "has_renderer": False,
            "has_offscreen_renderer": False,
            "use_camera_obs": False,
            "control_freq": 20,
            "horizon": max_timesteps,
            "reward_scale": 1.0,
            "random_force_point": current_curriculum[1],
            # "object_name": current_episode_object
        } 

        is_prismatic = False
        curriculum_env_kwargs['env_name'] = 'MultipleRevoluteEnv' 

        env_kwarg_list = [
            curriculum_env_kwargs for i in range(n_envs)
        ]
        for i in range(n_envs):
            env_kwarg_list[i]['object_name'] = random.choice(cur_curriculum_objects) if not isinstance(cur_curriculum_objects, str) else cur_curriculum_objects 
            print(f'process {i} is training with object {env_kwarg_list[i]["object_name"]}')
        env = SubprocVecEnv([make_env(curriculum_env_kwargs) for i in range(n_envs)])
        state = env.reset()
        cur_curriculum_training_steps = 0
        per_ep_projections = [[] for i in range(n_envs)]
        per_ep_rewards = [0 for i in range(n_envs)]
        cur_curriculum_projections = []
        cur_curriculum_rewards = []
        cur_curriculum_successes = []
        # cur_curriculum_noise_idx = 0
        training_start_time = time.time()
        while cur_curriculum_training_steps < episodes_schedule[curriculum_idx] * 200:
            action = policy.select_action(state)
            successes = env.get_attr('success')
            next_state, reward, done, _ = env.step(action)
            projections = env.get_attr('current_step_force_projection')
            for i in range(n_envs):
                replay_buffer.add((state[i], action[i], reward[i], next_state[i], float(done[i])))
                per_ep_rewards[i] += reward[i]
                per_ep_projections[i].append(projections[i])
                if done[i]:
                    cur_curriculum_projections.append(np.mean(per_ep_projections[i]))
                    per_ep_projections[i] = []
                    cur_curriculum_successes.append(successes[i])
                    cur_curriculum_rewards.append(per_ep_rewards[i])
                    per_ep_rewards[i] = 0
            cur_curriculum_training_steps += len(state)
            if cur_curriculum_training_steps % 100*200 == 0:
                cur_noise_idx = min(cur_noise_idx + 1, len(policy_noise_schedule)-1)
            if cur_curriculum_training_steps % 40 == 0:
                policy.update(replay_buffer, 30, batch_size, gamma, polyak, policy_noise_schedule[cur_noise_idx], noise_clip, policy_delay)
            if cur_curriculum_training_steps % 10000*24 == 0:
                policy.save(rev_force_policy_dir, f'curriculum_{curriculum_idx}_step_{cur_curriculum_training_steps}')
            if cur_curriculum_training_steps % 10000 == 0:
                print(f'Curriculum {curriculum_idx} training timesteps: {cur_curriculum_training_steps}')
                print(f'Total training timesteps: {current_training_timesteps + cur_curriculum_training_steps}')
                print(f'Curriculum {curriculum_idx} averate reward: {np.mean(cur_curriculum_rewards[-100:])}')
                print(f'Curriculum {curriculum_idx} current projection ratio: {np.mean(cur_curriculum_projections[-100:])}')
                print(f'Curriculum {curriculum_idx} current success rate: {np.mean(cur_curriculum_successes[-100:])}')
            if cur_curriculum_training_steps > 0.1*episodes_schedule[curriculum_idx] * 200 and np.mean(cur_curriculum_projections[-100]) > 0.85:
                print(f'Curriculum {curriculum_idx} reached the projection threshold in {cur_curriculum_training_steps} steps')
                break
        current_training_timesteps += cur_curriculum_training_steps
        training_end_time = time.time() 
        time_spent_on_training += training_end_time - training_start_time
        policy.save(rev_force_policy_dir, f'curriculum_{curriculum_idx}_final')
        env.close()
    policy.save(rev_force_policy_dir, f'revolute_final')
    json_dict = {
        'timesteps': current_training_timesteps,
        'time_spent': time_spent_on_training
    }
    with open(rev_force_policy_dir + '/training_time.json', 'w') as f:
        json.dump(json_dict, f)
        
        # for episodes in range(1, 1+episodes_schedule[curriculum_idx]):
        #     # env = ActionRepeatWrapperNew(raw_env, action_repeat) 
        #     cur_ep_rwds = [] 
        #     cur_ep_projections = []
            
        #     # state = env.reset()  
        #     # print('parallel environment state: ', state)
        #     # state = prismatic_state(state) if is_prismatic else revolute_state(state)
        #     done = False 

        #     if episodes % 20 == 0:
        #         cur_noise_idx = min(cur_noise_idx + 1, len(policy_noise_schedule)-1) 
            
        #     # policy = prismatic_policy if is_prismatic else revolute_policy
        #     # policy = TD3('MlpPolicy', env, verbose=1)
        #     policy = prismatic_policy if is_prismatic else revolute_policy
        #     replay_buffer = prismatic_replay_buffer if is_prismatic else revolute_replay_buffer 

        #     cur_ep_start_time = time.time()
            

        #     # training for each episodes
        #     for t in range(max_timesteps):
        #         action = policy.select_action(state)
        #         next_state, reward, done, _ = env.step(action)  
        #         # next_state = prismatic_state(next_state) if is_prismatic else revolute_state(next_state)
                
        #         cur_ep_rwds.append(reward) 
        #         # print(reward)
        #         for i in range(n_envs):
        #             replay_buffer.add((state[i], action[i], reward[i], next_state[i], float(done[i])))
        #         state = next_state
        #         if t==(max_timesteps-1) :
        #             # for i in range(n_envs):
        #             policy.update(replay_buffer, t, batch_size, gamma, polyak, policy_noise_schedule[cur_noise_idx], noise_clip, policy_delay)
        #             print(f'Episode {episodes}: rwd: {np.sum(cur_ep_rwds)/n_envs}') 
        #             # episode_success = env.success
        #             # add current episode data to the list
        #             cur_curriculum_rewards.append(np.sum(cur_ep_rwds)/n_envs)
        #             # cur_curriculum_successes.append(env._check_success()) # The task is considered success if reward >= 800.
        #             # cur_curriculum_projections.append(np.mean(cur_ep_projections))
        #             # env.close()
        #             # break
            
        #     cur_ep_end_time = time.time() 
        #     cur_ep_time_spent = cur_ep_end_time - cur_ep_start_time 
        #     time_spent_on_training += cur_ep_time_spent 

        #     with open(f'{log_dir}/physical_training_time.json', 'w') as f:
        #         json.dump(time_spent_on_training, f)

        #     cur_curriculum_latest_rwd = np.mean(cur_curriculum_rewards[max(-20, -episodes):]) 
        #     # cur_curriculum_latest_sr = np.mean(cur_curriculum_successes[max(-20, -episodes):])
        #     print(f'Curriculum {curriculum_idx} average reward (past 20 episode): {cur_curriculum_latest_rwd}')
            # print(f'Curriculum {curriculum_idx} success rate (past 20 episode): {cur_curriculum_latest_sr}')


        #     cur_curriculum_success_rates.append(np.mean(cur_curriculum_successes))
        #     if episodes % save_policy_freq == 0:
        #         # print('Saving checkpoint')
        #         if is_prismatic:
        #             policy.save(
        #                 directory=pris_force_policy_dir,
        #                 name=f"curriculum_{curriculum_idx}_ep_{episodes}")
        #         else:
        #             policy.save(
        #                 directory=rev_force_policy_dir,
        #                 name=f"curriculum_{curriculum_idx}_ep_{episodes}")

        #     # do the rollouts
        #     if episodes >= episodes_schedule[curriculum_idx] - 1: # Current curriculum is finished
        #         print(f'Curriculum {curriculum_idx} is finished training. Evaluating.') 

        #         for ep in range(rollouts):
        #             # make the environments  
        #             current_episode_object = random.choice(cur_curriculum_objects) if not isinstance(cur_curriculum_objects, str) else cur_curriculum_objects
        #             print(f'Evaluation episode {ep} is evaluating with object {current_episode_object}')
                    
        #             curriculum_env_kwargs = {
        #                 "init_object_angle": current_curriculum[0],
        #                 "has_renderer": False,
        #                 "has_offscreen_renderer": False,
        #                 "use_camera_obs": False,
        #                 "control_freq": 20,
        #                 "horizon": max_timesteps,
        #                 "reward_scale": 1.0,
        #                 "random_force_point": current_curriculum[1],
        #                 "object_name": current_episode_object
        #             }

        #             # Environment initialization
        #             raw_env = suite.make(
        #         "MultipleRevoluteEnv", 
        #         **curriculum_env_kwargs
        #         ) if current_episode_object not in ['train-drawer-1'] else suite.make(
        #             "TrainPrismaticEnv",
        #             **curriculum_env_kwargs
        #         )
                
        #             env = ActionRepeatWrapperNew(raw_env, action_repeat) 

        #             cur_ep_eval_ep_rwds = [] 
        #             cur_ep_eval_projections = []

        #             state = env.reset()

        #             state = prismatic_state(state) if is_prismatic else revolute_state(state)                 
        #             done = False
                    
        #             for t in range(max_timesteps):

        #                 action = policy.select_action(state)

        #                 next_state, reward, done, _ = env.step(action)
        #                 next_state = prismatic_state(next_state) if is_prismatic else revolute_state(next_state)
                         
        #                 action_projection = env.current_action_projection
        #                 cur_ep_eval_ep_rwds.append(reward)
        #                 cur_ep_eval_projections.append(action_projection) 
        #                 # env.render()

        #                 state = next_state

        #                 if done or t == max_timesteps-1 or env._check_success():

        #                     current_episode_success = env.success

        #                     cur_run_eval_rwds.append(np.sum(cur_ep_eval_ep_rwds)) 
        #                     cur_run_eval_projections.append(np.mean(cur_ep_eval_projections))
        #                     cur_run_eval_success_rates.append(current_episode_success)
        #                     env.close()
        #                     break
        #         print(f'Curriculum {curriculum_idx} gets {np.mean(cur_run_eval_rwds[-rollouts:])} average rewards per episode, {np.mean(cur_run_eval_success_rates[-rollouts:])} success rates')

        # cur_run_rewards.extend(cur_curriculum_rewards)
        # cur_run_projections.extend(cur_curriculum_projections)

        # with open(f'{log_dir}/reward_train_curriculum_{curriculum_idx}.json', 'w') as f:
        #     json.dump(cur_curriculum_rewards, f) 
        
        # with open(f'{log_dir}/success_rate_train_curriculum_{curriculum_idx}.json', 'w') as f:
        #     json.dump(cur_curriculum_success_rates, f) 

        # save the checkpoint
        if is_prismatic:
            policy.save(
                directory=pris_force_policy_dir,
                name=f"curriculum_{curriculum_idx}")
        else:
            policy.save(
                directory=rev_force_policy_dir,
                name=f"curriculum_{curriculum_idx}")

    # record the evaluation results
    with open(f'{log_dir}/reward_eval_{run_id}.json', 'w') as f:
        json.dump(cur_run_eval_rwds, f)
    with open(f'{log_dir}/projection_eval_{run_id}.json', 'w') as f:
        json.dump(cur_run_eval_projections, f)
    with open(f'{log_dir}/success_rate_eval_{run_id}.json', 'w') as f:
        json.dump(cur_run_eval_success_rates, f) 




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
    for i in range(1):
        train(0)
    
