import torch
from agent.td3 import TD3, ReplayBuffer
import numpy as np
import time
from env.wrappers import make_env
from env.train_prismatic_env import TrainPrismaticEnv
from stable_baselines3.common.vec_env import SubprocVecEnv


if __name__ == '__main__':
    n_envs = 24
    batch_size = 32
    max_timesteps = 200
    prismatic_noise_idx = 0
    prismatic_training_timesteps = 0
    prismatic_training_time = 0.
    per_ep_projections = [[] for i in range(n_envs)]
    per_ep_rewards = [0 for i in range(n_envs)]
    prismatic_rewards = []
    prismatic_projections = []
    prismatic_success = []
    gamma = 0.99 
    polyak = 0.95
    noise_clip = 2
    policy_delay = 2
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

    pris_force_policy_dir = 'checkpoints/il_policies'
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

    prismatic_policy = TD3(1e-3, 6, 3, 5)
    prismatic_policy.load('checkpoints/il_policies', 'prismatic_pretrained')
    prismatic_replay_buffer = ReplayBuffer()

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