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
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


controller_name = "OSC_POSE"
controller_cfg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"controller_configs")
controller_cfg_path = os.path.join(controller_cfg_dir, "osc_pose_absolute.json")
controller_configs = suite.load_controller_config(custom_fpath = controller_cfg_path,default_controller=controller_name)
with open(controller_cfg_path, 'r') as f:
    controller_configs = json.load(f)
env_kwargs = dict(
    robots="Panda",
    # robots="UR5e",
    object_name = "cabinet-1",
    # object_name = "trashcan-1",
    obj_rotation=(-np.pi/2, -np.pi/2),
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
    x_range = (0.5, 0.5),
    y_range = (0.4, 0.4), 
    cache_video=True
)
env_name = "RobotPrismaticEnv"
env:RobotPrismaticEnv = suite.make(
    env_name,
    **env_kwargs
)

obs = env.reset()
# set the open percentage
env.set_open_percentage(0.3)

robot_gripper_pos = obs["robot0_eef_pos"]

# get the joint qpos
reset_joint_qpos = env.sim.data.qpos[env.slider_qpos_addr]

sim_utils.init_camera_pose(env, camera_pos=np.array([-0.77542536, -0.02539806,  0.30146208]), scale_factor=3) 

# if want to visualize the point cloud in open3d, the environment later will not work
viz_imgs = False
need_o3d_viz = False
run_cgn = False

pcd_wf_path = f'point_clouds/world_frame_pointclouds/world_frame_{env_kwargs["object_name"]}.ply'
pcd_wf_no_downsample_path = f'point_clouds/world_frame_pointclouds/world_frame_{env_kwargs["object_name"]}_no_downsample.ply'
camera_info_path = f'infos/camera_info_{env_kwargs["object_name"]}.npz'


camera_euler = np.array([ 45., 22., 3.])
camera_euler_for_pos = np.array([camera_euler[-1], camera_euler[1], -camera_euler[0]])
camera_rotation = R.from_euler("xyz", camera_euler_for_pos, degrees=True)
camera_quat = R.from_euler("xyz", camera_euler, degrees=True).as_quat()

camera_forward = np.array([1, 0, 0])
world_forward = camera_rotation.apply(camera_forward)
distance = 1.5
camera_pos = -distance * world_forward

reset_x_range = (1,1)
reset_y_range = (1,1)

pcd_wf_path, pcd_wf_no_downsample_path,camera_info_path = get_aograsp_ply_and_config(env_name = env_name, 
                        env_kwargs=env_kwargs,
                        object_name=env_kwargs["object_name"], 
                        # camera_pos=np.array([-0.77542536, -0.02539806,  0.30146208]),
                        camera_pos=camera_pos,
                        camera_quat=camera_quat,
                        scale_factor=1,
                        pcd_wf_path=pcd_wf_path,
                        pcd_wf_no_downsample_path=pcd_wf_no_downsample_path,
                        camera_info_path=camera_info_path,
                        viz=viz_imgs, 
                        need_o3d_viz=need_o3d_viz,
                        reset_joint_qpos=reset_joint_qpos,
                        reset_x_range = reset_x_range, 
                        reset_y_range = reset_y_range,
                        )

# get the affordance heatmap
# pcd_cf_path, affordance_path = get_affordance_main(pcd_wf_path, camera_info_path, 
#                                                    viz=False) 

pcd_cf_path = 'point_clouds/camera_frame_pointclouds/camera_frame_cabinet-1.ply'
affordance_path = 'outputs/point_score/camera_frame_cabinet-1_affordance.npz'

# get the grasp proposals
world_frame_proposal_path, top_k_pos_wf, top_k_quat_wf = get_grasp_proposals_main(
    pcd_cf_path, 
    affordance_path, 
    camera_info_path, 
    run_cgn=run_cgn, 
    viz=need_o3d_viz, 
    save_wf_pointcloud=True,
    object_name=env_kwargs["object_name"],
    top_k=-1,
)

# get the joint parameters
joint_poses, joint_directions, joint_types, results_list = get_joint_param_main(
    pcd_wf_no_downsample_path, 
    camera_info_path, 
    viz=need_o3d_viz, 
    return_results_list=True)
object_pos = np.array(env.obj_pos)

# find the first joint in the list with joint_type = 1
joint_pose_selected = None
joint_direction_selected = None

for i in range(len(results_list)):
    if joint_types[i] == 1:
        joint_pose_selected = joint_poses[i]
        joint_direction_selected = joint_directions[i]
        break

# if not found, use the first joint
if joint_pose_selected is None:
    print(termcolor.colored("No joint with type 1 found, using the first joint", "red"))
    joint_pose_selected = joint_poses[0]
    joint_direction_selected = joint_directions[0]

# failed direction like this:[-6.41057711e-01 -8.01612643e-05  7.67492674e-01]
joint_direction_selected = joint_direction_selected / np.linalg.norm(joint_direction_selected)

print(termcolor.colored("joint pose selected: ", "green"), joint_pose_selected)
print(termcolor.colored("joint direction selected: ", "green"), joint_direction_selected)
joint_direction_selected = -joint_direction_selected


obs_dim = 6
action_dim = 3
gamma = 0.99 
lr = 0.001
lr_critic = 0.0001 
K_epochs = 4 
eps_clip = 0.2 
action_std = 0.5 
has_continuous_action_space = True
max_action = float(5)


# load the trained policy
policy_dir = 'checkpoints/force_policies/prismatic_td3/run_0/curriculum_5'
policy_name = 'prismatic_td3_0_curriculum_5'
policy = TD3(lr, obs_dim, action_dim, max_action)
policy.load(policy_dir, policy_name)


# world frame proposals need to be offset by the object's position
# print("object pos: ", env.obj_pos)
object_pos = np.array(env.obj_pos)

# scale the proposals
# print("raw grasp pose: ", top_k_pos_wf[0])

# print("scaled grasp pose: ", top_k_pos_wf[0])

top_k_pos_wf = top_k_pos_wf + object_pos

# top_k_pos_wf = top_k_pos_wf + np.array([0, 0.1, 0.])

grasp_pos = top_k_pos_wf[1]
grasp_quat = top_k_quat_wf[1]

# print("Grasp Pos: ", grasp_pos)

# change quat to euler
sci_rotation = R.from_quat(grasp_quat)
further_rotation = R.from_euler('z', 90, degrees=True)
sci_rotation = sci_rotation * further_rotation
rotation_vector = sci_rotation.as_rotvec()

mid_point_pos = (grasp_pos + robot_gripper_pos) / 2
prepaer_grasp_pos = grasp_pos + np.array([-0., 0, 0.1])

action = np.concatenate([robot_gripper_pos, rotation_vector, [-1]])

for i in range(80):
    action = np.concatenate([prepaer_grasp_pos, rotation_vector, [-1]])
    env.step(action)
    # env.render()

final_grasp_pos = grasp_pos + np.array([0, 0, -0.05])

for i in range(50):
    action = np.concatenate([final_grasp_pos, rotation_vector, [-1]])
    env.step(action)
    # env.render()

# close the gripper
for i in range(50):
    action = np.concatenate([final_grasp_pos, rotation_vector, [1]])
    env.step(action)
    # env.render()

done = False
last_grasp_pos = final_grasp_pos

obs = np.concatenate([joint_direction_selected, np.array(obs["robot0_eef_pos"]) - final_grasp_pos])


# Interaction section

# Shake the object 
trajectory = [last_grasp_pos ]

for i in range(40):
    action = policy.select_action(obs)
    action = -action * 0.01

    action = np.concatenate([last_grasp_pos + action, rotation_vector, [1]])
    next_obs, reward, done, _ = env.step(action)
    last_grasp_pos = next_obs['robot0_eef_pos'] 
    trajectory.append(last_grasp_pos)
    next_obs = np.concatenate([joint_direction_selected, np.array(next_obs["robot0_eef_pos"]) - final_grasp_pos])
    obs = next_obs

for i in range(40):
    action = policy.select_action(obs)
    action = action * 0.01

    action = np.concatenate([last_grasp_pos + action, rotation_vector, [1]])
    next_obs, reward, done, _ = env.step(action)
    last_grasp_pos = next_obs['robot0_eef_pos'] 
    trajectory.append(last_grasp_pos)
    next_obs = np.concatenate([joint_direction_selected, np.array(next_obs["robot0_eef_pos"]) - final_grasp_pos])
    obs = next_obs

for i in range(40):
    action = policy.select_action(obs)
    action = -action * 0.01

    action = np.concatenate([last_grasp_pos + action, rotation_vector, [1]])
    next_obs, reward, done, _ = env.step(action)
    last_grasp_pos = next_obs['robot0_eef_pos'] 
    trajectory.append(last_grasp_pos)
    next_obs = np.concatenate([joint_direction_selected, np.array(next_obs["robot0_eef_pos"]) - final_grasp_pos])
    obs = next_obs



## Interaction finished

for i in range(200):
    action = policy.select_action(obs)
    action = action * 0.01

    action = np.concatenate([last_grasp_pos + action,rotation_vector, [1]])
    next_obs, reward, done, _ = env.step(action)
    last_grasp_pos = next_obs['robot0_eef_pos']
    next_obs = np.concatenate([joint_direction_selected, np.array(next_obs["robot0_eef_pos"]) - final_grasp_pos])

    obs = next_obs
    # env.render()

env.save_video('videos/robot_prismatic_shake.mp4')
env.close()


