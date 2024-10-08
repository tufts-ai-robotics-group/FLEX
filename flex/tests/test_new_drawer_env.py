import robosuite as suite
from env.train_prismatic_env import TrainPrismaticEnv
import numpy as np
import time

env:TrainPrismaticEnv = suite.make(
    "TrainPrismaticEnv",
    object_name = "train-drawer-1",
     random_force_point = True,
    # init_door_angle = (-np.pi + np.pi/4, -np.pi + np.pi/4),
    init_object_angle = (-np.pi,-np.pi),
    has_renderer=True,
    use_camera_obs=True,
    has_offscreen_renderer=True,
    camera_depths = True,
    camera_segmentations = "element",
    horizon=1000,
    camera_names = ["agentview", "sideview", "frontview", "birdview"],
    camera_heights = [1024,1024, 1024, 1024],
    camera_widths = [1024, 256, 1024, 1024],
    render_camera = "sideview",
)

obs = env.reset()
env.render()
for _ in range(1000):
    action = np.array([5,0.0,0.0])
    obs, _,_,_ = env.step(action)
    print(obs["joint_qpos"])
    env.render()
    # time.sleep(0.1)