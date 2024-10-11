# open drawer using robot
# TODO: use ikflow to get ik solution

import open3d as o3d
import robosuite as suite
import scipy.spatial
from env.robot_drawer_opening import RobotDrawerOpening
from env.drawer_opening import DrawerOpeningEnv
from utils.sim_utils import get_pointcloud, flip_image
from utils.gpd_utils import gpd_get_grasp_pose
import numpy as np
import matplotlib.pyplot as plt
from utils.sim_utils import initialize_robot_position
import robosuite.utils.transform_utils as transform
from utils.control_utils import PathPlanner, Linear, Gaussian

import os
import json

import scipy
from agent.ppo import PPO

# os.environ['MUJOCO_GL'] = 'osmesa'
# initialize robot controller
controller_name = "OSC_POSE"
controller_cfg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"controller_configs")
controller_cfg_path = os.path.join(controller_cfg_dir, "osc_pose_absolute.json")
controller_configs = suite.load_controller_config(custom_fpath = controller_cfg_path,default_controller=controller_name)

with open(controller_cfg_path, 'r') as f:
    controller_configs = json.load(f)
print(controller_configs)
env:RobotDrawerOpening = suite.make(
    "RobotDrawerOpening",
    robots="Panda",
    has_renderer=True,
    use_camera_obs=True,
    has_offscreen_renderer=True,
    camera_depths = True,
    camera_segmentations = "element",
    controller_configs=controller_configs,
    control_freq = 20,
    horizon=10000,
    camera_names = ["birdview", "agentview", "frontview", "sideview"],
    camera_heights = [1024,256,512,1024],
    camera_widths = [1024,1024,1024,1024]
)

# env:RobotDrawerOpening = suite.make(
#     "DrawerOpeningEnv",
#     has_renderer=True,
#     use_camera_obs=True,
#     has_offscreen_renderer=True,
#     camera_depths = True,
#     camera_segmentations = "element",
#     control_freq = 20,
#     horizon=10000,
#     camera_names = ["birdview", "agentview", "frontview", "sideview"],
#     camera_heights = [1024,1024,512,1024],
#     camera_widths = [1024,1024,1024,1024]
# )

obs = env.reset()
print(type(obs["gripper_quat"]))
# env.render()
camera_name = "frontview"
depth_image = obs['{}_depth'.format(camera_name)]
depth_image = flip_image(depth_image)
# plt.imshow(depth_image)
# plt.show()
# pointcloud = get_pointcloud(env, obs, ["agentview", "frontview","sideview"], [256,512,1024], [1024,1024,1024], ["handle", "foo",])
# o3d.visualization.draw_geometries([pointcloud])
# o3d.io.write_point_cloud("point_clouds/drawer_pointcloud.pcd", pointcloud)
# x = gpd_get_grasp_pose('drawer_pointcloud.pcd',cfg_file_name='experiment_grasp.cfg')
# for i in range(x.size):
#     print("=========================")
#     print(f"pose_{i}: {x.pos_x[i]}, {x.pos_y[i]}, {x.pos_z[i]}")
#     print(f"rotation matrix_{i}: {x.rotation_matrix_list[i][0]}, {x.rotation_matrix_list[i][1]},{x.rotation_matrix_list[i][2]}\
# ,{x.rotation_matrix_list[i][3]}, {x.rotation_matrix_list[i][4]},{x.rotation_matrix_list[i][5]}\
# ,{x.rotation_matrix_list[i][6]}, {x.rotation_matrix_list[i][7]},{x.rotation_matrix_list[i][8]}")

# This is from GPD and set still for further testing
# hand_pose = np.array([-0.176013,-0.632732,1.25141])
hand_pose = np.array([-0.2030102014541626, -0.540092134475708, 1.2524129152297974])
# Robosuite have an x pointing out of screen to you and an Z facing upward 
# rotation_euler = np.array([0.005762052722275257, -0.984636664390564,0.17452049255371094])
rotation_euler = np.array([0.005762052722275257, -0.984636664390564,0.17452049255371094])
# rotation_euler = np.array([0,0,0])
rotation_matrix = transform.euler2mat(rotation_euler)

rotation_matrix = np.array([[-0.0073337797075510025, -0.5499632358551025,0.8351566791534424],
    [0.0029766582883894444, -0.8351874351501465,-0.5499573349952698],          
    [0.9999686479568481, -0.001547289895825088,0.00776213314384222]])
rotation_matrix = np.array([[-0.10221315920352936, -0.9264970421791077,-0.3621542751789093],
                            [-0.3834446370601654, 0.37262317538261414,-0.845057487487793],
                            [0.9178903102874756, 0.05249011516571045,-0.3933473229408264]])
# rotation_matrix = np.transpose(rotation_matrix)

sci_rotation = scipy.spatial.transform.Rotation.from_matrix(rotation_matrix)
rotation_axis_angle = sci_rotation.as_rotvec()

# target_quat = transform.mat2quat(rotation_matrix)
# obs = env.reset()
# env.render()
# obs = initialize_robot_position(env, hand_pose)
target_euler = transform.mat2euler(rotation_matrix)

state_dim = 7
action_dim = 3
gamma = 0.99 
lr_actor = 0.0003 
lr_critic = 0.0001 
K_epochs = 4 
eps_clip = 0.2 
action_std = 0.5 
has_continuous_action_space = True 

ppo_agent = PPO( 
    state_dim, 
    action_dim, 
    lr_actor, 
    lr_critic, 
    gamma, 
    K_epochs, 
    eps_clip, 
    has_continuous_action_space, 
    action_std 
    )  

# model_pth = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"checkpoint_1100.pth")
# ppo_agent.load(model_pth)

action = np.concatenate([hand_pose, rotation_axis_angle,[-1]])

for i in range(80):
    obs = env.step(action)
    env.render()


hand_pose_new = np.array([-0.2030102014541626, -0.660092134475708, 1.2524129152297974])
action = np.concatenate([hand_pose_new, rotation_axis_angle,[0]])
for i in range(50):
    obs = env.step(action)
    env.render()

action = np.concatenate([hand_pose_new, rotation_axis_angle,[1]])
for i in range(10):
    obs = env.step(action)
    env.render()

action = np.concatenate([hand_pose_new, rotation_axis_angle,[1]])
obs, _,_,_ = env.step(action)
handle_pos = obs["handle_pos"]
handle_quat = obs["handle_quat"]
gripper_pos = obs["gripper_pos"]

for i in range(50):
    action = ppo_agent.select_action(np.concatenate([handle_pos, handle_quat]))
    action = action * 0.1
    # print(action)
    action_absolute = np.concatenate([gripper_pos + action, rotation_axis_angle, [1]])
    print(action_absolute)
    for j in range(4):
        obs, _,_,_ = env.step(action_absolute)
        env.render()
    handle_pos = obs["handle_pos"]
    handle_quat = obs["handle_quat"]
    gripper_pos = obs["gripper_pos"]

env.close()