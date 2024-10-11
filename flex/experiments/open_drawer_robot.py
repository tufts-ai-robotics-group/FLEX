# open drawer using robot
# TODO: use ikflow to get ik solution

import open3d as o3d
import robosuite as suite
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

# initialize robot controller


controller_name = "OSC_POSE"
controller_configs = suite.load_controller_config( default_controller=controller_name)
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
#     print(f"pose_{i}: {x.pos_x[0]}, {x.pos_y[0]}, {x.pos_z[0]}")
#     print(f"rotation matrix_{i}: {x.rotation_matrix_list[0][0]}, {x.rotation_matrix_list[0][1]},{x.rotation_matrix_list[0][2]}")

# This is from GPD and set still for further testing
hand_pose = np.array([-0.176013,-0.632732,1.25141])
# Robosuite have an x pointing out of screen to you and an Z facing upward 
rotation_euler = np.array([0.005762052722275257, -0.984636664390564, 0.17452049255371094])
rotation_matrix = transform.euler2mat(rotation_euler)
target_quat = transform.mat2quat(rotation_matrix)
# obs = env.reset()
# env.render()
# obs = initialize_robot_position(env, hand_pose)


# generate traj
# profile class required by the abr_control

start_rot_matrix = transform.quat2mat(obs["gripper_quat"])
start_euler = transform.mat2euler(start_rot_matrix)

planner = PathPlanner(Linear(), Gaussian(dt=0.05, acceleration=0.1))
path = planner.generate_path(
    start_position=obs["gripper_pos"],
    target_position=hand_pose,
    max_velocity=0.3,
    start_orientation=start_euler,
    target_orientation=rotation_euler,
    plot=False
)

last_pos=obs["gripper_pos"]
last_rot_matrix = start_rot_matrix
for i in range(len(path)):
    cur_location = path[i]
    cur_pos = cur_location[:3]
    cur_rot_matrix = cur_location[3:].reshape([3,3])
    rel_rot_matrix = np.matmul(transform.matrix_inverse(last_rot_matrix),cur_rot_matrix)
    pos_difference = cur_pos - last_pos
    rel_quat = transform.mat2quat(rel_rot_matrix)
    action = np.concatenate([pos_difference, rel_quat], axis=0)
    env.step(action)
    env.render()
    # print(rel_rot_matrix)
    last_rot_matrix = cur_rot_matrix
    last_pos = cur_pos
# for i in range(1000):
#        action = np.concatenate([hand_pose, target_quat],axis=0)
#        env.step(action)
#        env.render()

print(path.shape)
    
