from robosuite.wrappers import Wrapper
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import threading
from multiprocessing import Pool
import queue 
# from robosuite.wrappers.gym_wrapper import GymWrapper 
import robosuite as suite
from stable_baselines3.common.vec_env import SubprocVecEnv 
import gymnasium as gym 
from gymnasium import spaces, Env


class ActionRepeatWrapperNew(Wrapper):
    def __init__(self, env, action_repeat):
        super().__init__(env)
        self.action_repeat = action_repeat

    def step(self, action): 
        rwd = 0
        for _ in range(self.action_repeat):
            n_state, n_rwd, done, info = self.env.step(action)
            rwd += n_rwd
            if done:
                break
        return n_state, rwd, done, info
    

class MultiThreadingParallelEnvsWrapper:

    def __init__(self, envs: list):
        super().__init__() 
        self.envs = envs
        for(env) in self.envs:
            print(env)
        self.num_envs = len(envs)
        self.executor = ThreadPoolExecutor(max_workers=self.num_envs)

    def reset(self):
        # with ThreadPoolExecutor(max_workers=self.num_envs) as executor:
        #     futures = [executor.submit(self._reset, env) for env in self.envs]
        #     results = [future.result() for future in as_completed(futures)]
        futures = [self.executor.submit(self._reset, env) for env in self.envs]
        results = [future.result() for future in as_completed(futures)]
        
        return results
    
    def _reset(self, env):
        # print(f"Executed in thread: {threading.current_thread().name}")
        return env.reset()

    def step(self, actions):
        # with ThreadPoolExecutor(max_workers=self.num_envs) as executor:
            # futures = [executor.submit(self._step_env, env, action) for env, action in zip (self.envs, actions)]
            # results = [future.result() for future in as_completed(futures)] 
        futures = [self.executor.submit(self._step_env, env, action) for env, action in zip (self.envs, actions)]
        results = [future.result() for future in as_completed(futures)] 
        # with Pool(self.num_envs) as pool:
        #     results = pool.starmap(self._step_env, zip(self.envs, actions))
        obs = np.array([result[0] for result in results]) 
        rewards = np.array([result[1] for result in results])
        dones = np.array([result[2] for result in results])
        infos = [result[3] for result in results]
        return obs, rewards, dones, infos 
    
    def _step_env(self, env, action):
        # print(f"Executed in thread: {threading.current_thread().name}")
        return env.step(action)

    def render(self, mode='human'):
        with ThreadPoolExecutor(max_workers=self.num_envs) as executor:
            futures = [executor.submit(env.render, mode) for env in self.envs]
            results = [future.result() for future in as_completed(futures)]
        return results

    def close(self):
        for env in self.envs:
            env.close()

class MultiThreadingParallelEnvsWrapperNew:
    def __init__(self, envs):
        # self.env_name = env_name
        # self.num_envs = num_envs
        # self.env_kwargs = env_kwargs
        # self.envs = [self._create_env() for _ in range(num_envs)]
        self.envs = envs
        num_envs = len(envs)
        self.num_envs = num_envs
        
        # Create a queue for each environment
        self.task_queues = [queue.Queue() for _ in range(num_envs)]
        
        # Start worker threads for each environment
        self.threads = []
        for i in range(num_envs):
            thread = threading.Thread(target=self._worker, args=(self.envs[i], self.task_queues[i]))
            thread.daemon = True  # Daemon threads exit when the main thread does
            thread.start()
            self.threads.append(thread)

    def _worker(self, env, task_queue):
        while True:
            # Get the next task
            func, future, args = task_queue.get()
            if func is None:
                break  # Exit signal
            try:
                result = func(env, *args)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
            task_queue.task_done()

    def _submit_task(self, func, env_index, *args):
        future = Future()
        self.task_queues[env_index].put((func, future, args))
        return future

    def reset(self):
        futures = [self._submit_task(self._reset_env, i) for i in range(self.num_envs)]
        return [f.result() for f in futures]

    def _reset_env(self, env):
        return env.reset()

    def step(self, actions):
        futures = [self._submit_task(self._step_env, i, action) for i, action in enumerate(actions)]
        results = [f.result() for f in futures] 
        obs = np.array([result[0] for result in results])
        rewards = np.array([result[1] for result in results])
        dones = np.array([result[2] for result in results])
        infos = [result[3] for result in results]
        return obs, rewards, dones, infos

    def _step_env(self, env, action):
        return env.step(action)

    def render(self, mode='human'):
        futures = [self._submit_task(self._render_env, i, mode) for i in range(self.num_envs)]
        return [f.result() for f in futures]

    def _render_env(self, env, mode):
        return env.render(mode)

    def close(self):
        # Signal the workers to exit and wait for them to finish
        for task_queue in self.task_queues:
            task_queue.put((None, None, None))  # Send exit signal
        for thread in self.threads:
            thread.join()

        # Close environments
        for env in self.envs:
            env.close()

## Multiprocessing with sb3 
"""
This file implements a wrapper for facilitating compatibility with OpenAI gym.
This is useful when using these environments with code that assumes a gym-like
interface.
"""

class GymWrapper(Wrapper, gym.Env):
    metadata = None
    render_mode = None
    def __init__(self, env, keys=None):
        super().__init__(env=env)
        # robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
        # self.name = robots + "_" + type(self.env).__name__

        # Get reward range
        self.reward_range = (0, self.env.reward_scale)
        self.is_prismatic = env.is_prismatic

        if keys is None:
            keys = [] 
            if self.is_prismatic:
                keys += ["joint_direction", "force_point_relative_to_start"]
            else:
                keys += ["hinge_direction", "force_point", "hinge_position"]
            # Add object obs if requested
            # if self.env.use_object_obs:
            #     keys += ["object-state"]
            # # Add image obs if requested
            # if self.env.use_camera_obs:
            #     keys += [f"{cam_name}_image" for cam_name in self.env.camera_names]
            # Iterate over all robots to add to state
            # for idx in range(len(self.env.robots)):
            #     keys += ["robot{}_proprio-state".format(idx)]
        self.keys = keys
        self.is_prismatic = env.is_prismatic

        # Gym specific attributes
        self.env.spec = None

        # set up observation and action spaces
        obs = self.env.reset()
        self.modality_dims = {key: obs[key].shape for key in self.keys}
        flat_ob = self._flatten_obs(obs)
        self.obs_dim = flat_ob.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high) 
        # print('action spec: ', self.env.action_spec)
        low, high = self.env.action_spec
        self.action_space = spaces.Box(low, high)

    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.

        Args:
            obs_dict (OrderedDict): ordered dictionary of observations
            verbose (bool): Whether to print out to console as observation keys are processed

        Returns:
            np.array: observations flattened into a 1d array
        """
        ob_lst = []
        for key in self.keys:
            if key in obs_dict:
                if verbose:
                    print("adding key: {}".format(key))
                ob_lst.append(np.array(obs_dict[key]).flatten()) 

        if self.is_prismatic:
            ob_list = [obs_dict["joint_direction"], obs_dict["force_point_relative_to_start"]]
        else:
            ob_list = [obs_dict['hinge_direction'], 
                        obs_dict['force_point']-obs_dict['hinge_position']-np.dot(obs_dict['force_point']-obs_dict['hinge_position'], obs_dict['hinge_direction'])*obs_dict['hinge_direction']
                        ] 
            
        # print(ob_list)
        return np.concatenate(ob_list)

    def reset(self, seed=None, options=None):
        """
        Extends env reset method to return flattened observation instead of normal OrderedDict and optionally resets seed

        Returns:
            np.array: Flattened environment observation space after reset occurs
        """
        if seed is not None:
            if isinstance(seed, int):
                np.random.seed(seed)
            else:
                raise TypeError("Seed must be an integer type!")
        ob_dict = self.env.reset()
        return self._flatten_obs(ob_dict), {}

    def step(self, action):
        """
        Extends vanilla step() function call to return flattened observation instead of normal OrderedDict.

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (np.array) flattened observations from the environment
                - (float) reward from the environment
                - (bool) episode ending after reaching an env terminal state
                - (bool) episode ending after an externally defined condition
                - (dict) misc information
        """
        ob_dict, reward, terminated, info = self.env.step(action) 
        terminated = terminated or self.env._check_success()
        return self._flatten_obs(ob_dict), reward, terminated, False, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Dummy function to be compatible with gym interface that simply returns environment reward

        Args:
            achieved_goal: [NOT USED]
            desired_goal: [NOT USED]
            info: [NOT USED]

        Returns:
            float: environment reward
        """
        # Dummy args used to mimic Wrapper interface
        return self.env.reward()
    
    def _check_success(self):
        """
        Dummy function to be compatible with gym interface that simply returns environment success

        Returns:
            bool: environment success
        """
        return self.env._check_success()
    

class SubprocessEnv(gym.Wrapper):
    def __init__(self, env_kwargs):
        self.env = suite.make(**env_kwargs)
        self.env = GymWrapper(self.env) 
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, seed=0):
        return self.env.reset(seed)

    def step(self, action):
        return self.env.step(action)

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        self.env.close()


def make_env(env_kwargs):
    def _init():
        # MuJoCo-related objects should be created here
        env = SubprocessEnv(env_kwargs)
        # env.seed(seed)
        return env
    return _init

