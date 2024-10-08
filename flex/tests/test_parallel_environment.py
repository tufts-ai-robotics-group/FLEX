import robosuite as suite 
import numpy as np
from env.wrappers import ActionRepeatWrapperNew, MultiThreadingParallelEnvsWrapper, MultiThreadingParallelEnvsWrapperNew, MultiProcessingParallelEnvsWrappeer
from env.train_multiple_revolute_env import MultipleRevoluteEnv  
from env.train_prismatic_env import TrainPrismaticEnv 
from env.robot_prismatic_env import RobotPrismaticEnv
from agent.td3 import TD3
import time 
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv


if __name__ == "__main__":

    n_envs = 10
    env_kwargs = {
                    'env_name': "TrainPrismaticEnv",
                    "init_object_angle": (-np.pi/2, np.pi/2),
                    "has_renderer": False,
                    "has_offscreen_renderer": False,
                    "use_camera_obs": False,
                    "control_freq": 20,
                    "horizon": 1000,
                    "reward_scale": 1.0,
                    "random_force_point": True,
                    # "object_name": 'train-cabinet-1'
                } 

    # envs = [suite.make('TrainPrismaticEnv', **env_kwargs) for _ in range(n_envs)] 

    env = MultiProcessingParallelEnvsWrappeer(n_envs=n_envs, env_kwargs=env_kwargs)
    env = Vector
    state_dim, action_dim = 6, 3
    max_action = 5.0 
    lr = 0.0025
    policy = TD3(lr, state_dim, action_dim, max_action) 

    # env = MultiThreadingParallelEnvsWrapperNew(envs)
    obs = env.reset()
    start_time = time.time()
    action = np.zeros((n_envs, action_dim))
    rwds = []
    for i in range(10):
        for j in range(1000):
            obs, rwd, done, _ = env.step(action) 
            rwds.extend(rwd.tolist())
        # obs = env.reset()
    # print(obs)
    end_time = time.time() 
    print(f'Gathered {len(rwds)} samples, time taken: ', end_time - start_time)
    env.close()

    env = suite.make(**env_kwargs) 
    obs = env.reset() 

    action = np.zeros(action_dim)
    rwds = []
    start_time = time.time()
    for i in range(10 * n_envs):
        for j in range(1000):
            obs, rwd, done, _ = env.step(action) 
            rwds.append(rwd)
        obs = env.reset()
    end_time = time.time()
    print(f'Single thread gathered {len(rwds)} samples, time taken: ', end_time - start_time)
    env.close()
    # print(type(obs))

    # if __name__ == "__main__":
    #     num_tasks = 5

    #     with ThreadPoolExecutor(max_workers=num_tasks) as executor:
    #         futures = [executor.submit(test_function, i) for i in range(num_tasks)]
            
    #         # Optionally, wait for all tasks to complete
    #         for future in futures:
    #             future.result()  # This will block until the individual task is done
