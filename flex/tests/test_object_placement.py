from env.robot_prismatic_env import RobotPrismaticEnv 
import open3d as o3d
import robosuite as suite
from utils.sim_utils import get_pointcloud, flip_image
from utils.gpd_utils import gpd_get_grasp_pose
import numpy as np
import matplotlib.pyplot as plt
from utils.sim_utils import initialize_robot_position
import robosuite.utils.transform_utils as transform
from utils.control_utils import PathPlanner, Linear, Gaussian
import os 
import json
from scipy.spatial.transform import Rotation as R 
import utils.sim_utils as sim_utils
from grasps.aograsp.get_pointclouds import get_aograsp_ply_and_config
from grasps.aograsp.get_affordance import get_affordance_main
from grasps.aograsp.get_proposals import get_grasp_proposals_main
from gamma.get_joint_param import get_joint_param_main
from agent.td3 import TD3
from termcolor import colored, cprint
import termcolor

controller_name = "OSC_POSE"

controller_cfg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"controller_configs")
controller_cfg_path = os.path.join(controller_cfg_dir, "osc_pose_absolute.json")
controller_configs = suite.load_controller_config(custom_fpath = controller_cfg_path,default_controller=controller_name)

with open(controller_cfg_path, 'r') as f:
    controller_configs = json.load(f)

env_kwargs = dict(
    robots="Panda",
    # robots="UR5e",
    object_name = "cabinet-3",
    # object_name = "trashcan-1",
    scale_object = True,
    object_scale = 0.5,
    has_renderer=True,
    use_camera_obs=True,
    has_offscreen_renderer=True,
    camera_depths = True,
    camera_segmentations = "element",
    controller_configs=controller_configs,
    control_freq = 20,
    horizon=10000,
    camera_names = ["birdview", "agentview", "frontview", "sideview"],
    # render_camera = "birdview",
    camera_heights = [1024,256,512,1024],
    camera_widths = [1024,1024,1024,1024],
    move_robot_away = False,

    obj_rotation = (-np.pi/2, -np.pi/2),
    rotate_around_robot = True,
    object_robot_distance = (0.707, 0.707),
    # y_range = (0.5, 0.5),
    # y_range = (-2, -2),
)

env_name = "RobotPrismaticEnv"

env:RobotPrismaticEnv = suite.make(
    env_name,
    **env_kwargs
)

action = np.zeros(7)
for _ in range(10):
    env.reset()
    env.render()
    for _ in range(10):
        env.step(action)
        env.render()