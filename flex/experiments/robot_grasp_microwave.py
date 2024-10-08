from env.robot_revolute_env import RobotRevoluteOpening 
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
import json
from scipy.spatial.transform import Rotation as R 
import utils.sim_utils as sim_utils
from grasps.aograsp.get_pointclouds import get_aograsp_ply_and_config
from grasps.aograsp.get_affordance import get_affordance_main
from grasps.aograsp.get_proposals import get_grasp_proposals_main
from gamma.get_joint_param import get_joint_param_main
from agent.td3 import TD3


object_name = "temp_door" # A microwave called "drawer"!!!

controller_name = "OSC_POSE"
controller_cfg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"controller_configs")
controller_cfg_path = os.path.join(controller_cfg_dir, "osc_pose_absolute.json")
controller_configs = suite.load_controller_config(custom_fpath = controller_cfg_path,default_controller=controller_name)

with open(controller_cfg_path, 'r') as f:
    controller_configs = json.load(f)
# print(controller_configs)


env_kwargs = dict(
    robots="Panda",
    # robots="Kinova3",
    object_type = "microwave",
    # object_model_idx = 2,
    obj_rotation=(-np.pi/2, -np.pi/2),
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
    x_range = (0.8,0.8),
    y_range = (0.4, 0.4),
    # y_range = (-2, -2),
)
env_name = "RobotRevoluteOpening"

env:RobotDrawerOpening = suite.make(
    env_name,
    **env_kwargs
)

obs = env.reset()

reset_joint_qpos = np.pi/6
# # set the joint qpos of the microwave
env.sim.data.qpos[env.slider_qpos_addr] = np.pi/6
env.sim.forward()

robot_gripper_pos = obs["robot0_eef_pos"]

sim_utils.init_camera_pose(env, camera_pos=np.array([-0.77542536, -0.02539806,  0.30146208]), scale_factor=3) 

# if want to visualize the point cloud in open3d, the environment later will not work
need_o3d_viz = True

pcd_wf_path, pcd_wf_no_downsample_path,camera_info_path = get_aograsp_ply_and_config(env_name = env_name, 
                        env_kwargs=env_kwargs,
                        object_name=env_kwargs["object_type"], 
                        camera_pos=np.array([-0.77542536, -0.02539806,  0.30146208]),
                        camera_quat=np.array([-0.005696068282031459, 0.19181093598117868, 0.02913152799094475, 0.9809829120433564]),
                        scale_factor=1,
                        pcd_wf_path='point_clouds/world_frame_pointclouds/world_frame_temp_door.ply',
                        pcd_wf_no_downsample_path='point_clouds/world_frame_pointclouds/world_frame_temp_door_no_downsample.ply',
                        viz=False, 
                        need_o3d_viz=need_o3d_viz,
                        reset_joint_qpos=reset_joint_qpos,
                        )

print(pcd_wf_path)
print(camera_info_path)

# get the affordance heatmap
pcd_cf_path, affordance_path = get_affordance_main(pcd_wf_path, camera_info_path)


# get the grasp proposals
world_frame_proposal_path, top_k_pos_wf, top_k_quat_wf = get_grasp_proposals_main(pcd_cf_path, affordance_path, camera_info_path, run_cgn=False, viz=need_o3d_viz, save_wf_pointcloud=True)


# get the joint parameters
joint_poses, joint_directions, joint_types = get_joint_param_main(pcd_wf_no_downsample_path, camera_info_path, viz=need_o3d_viz)
object_pos = np.array(env.obj_pos)
if len(joint_poses ) >1:
    joint_pose = joint_poses[0]
    joint_direction = joint_directions[0]
else:
    joint_pose = joint_poses[0]
    joint_direction = joint_directions[0]
joint_poses += object_pos
joint_pose += object_pos
joint_direction = joint_direction / np.linalg.norm(joint_direction)
print("joint_poses: ", joint_poses)
print("joint_directions: ", joint_directions)
print("joint_types: ", joint_types)
print("joint pose", joint_pose)


# print(type(obs["gripper_quat"]))
# env.render()
# camera_name = "frontview"
# depth_image = obs['{}_depth'.format(camera_name)]
# depth_image = flip_image(depth_image)

# proposals = np.load(f'outputs/grasp_proposals/world_frame_proposals/world_frame_{object_name}_grasp.npz', allow_pickle=True)
# print(proposals.keys())

# proposal_pos = proposals['pos'] # np.array(np.array(3), ...)
# proposal_quat = proposals['quat'] # np.array(np.array(4), ...)
# proposal_scores = proposals['scores'] # np.array(np.float32, ...)

# print(type(proposal_pos[0])) 
# print(type(proposal_scores[0])) 




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
policy_dir = 'checkpoints/force_policies'
policy_name = 'curriculum_door_continuous_random_point_td3_0_curriculum_8'
policy = TD3(lr, obs_dim, action_dim, max_action)
policy.load(policy_dir, policy_name)


# world frame proposals need to be offset by the object's position
print("object pos: ", env.obj_pos)
object_pos = np.array(env.obj_pos)
# object_pos = np.array([0.2, -1.,   0.9])

# scale the proposals
print("raw grasp pose: ", top_k_pos_wf[0])
# top_k_pos_wf = np.array(top_k_pos_wf) * 3

print("scaled grasp pose: ", top_k_pos_wf[0])

top_k_pos_wf = top_k_pos_wf + object_pos

# top_k_pos_wf = top_k_pos_wf + np.array([0, 0.1, 0.])

grasp_pos = top_k_pos_wf[0]
grasp_quat = top_k_quat_wf[0]

print("Grasp Pos: ", grasp_pos)

# change quat to euler
sci_rotation = R.from_quat(grasp_quat)
rotation_vector = sci_rotation.as_rotvec()

mid_point_pos = (grasp_pos + robot_gripper_pos) / 2
prepaer_grasp_pos = grasp_pos + np.array([-0.1, 0, 0.1])

action = np.concatenate([robot_gripper_pos, rotation_vector, [-1]])

for i in range(80):
    # action = np.zeros_like(env.action_spec[0])
    action = np.concatenate([prepaer_grasp_pos, rotation_vector, [-1]])
    env.step(action)
    env.render()


# rortation_correction = np.array([-0.5, 0, 0])

# rotation_original = sci_rotation.as_matrix()
# rotation_correction = R.from_euler("z", np.pi/6).as_matrix()

# rotation_final = rotation_correction @ rotation_original
# rotation_final = R.from_matrix(rotation_final).as_rotvec()


# grasp_pos -= np.array([0, 0.2, 0.])
# grasp_pos += np.array([0, -0.01, 0.05])
final_grasp_pos = grasp_pos + np.array([0, 0, -0.05])

for i in range(80):
    # action = np.zeros_like(env.action_spec[0])
    action = np.concatenate([final_grasp_pos, rotation_vector, [-1]])
    env.step(action)
    env.render()

# close the gripper
for i in range(80):
    action = np.concatenate([final_grasp_pos, rotation_vector, [1]])
    env.step(action)
    env.render()

# move the robot according to the policy

# use the ground truth hinge position and direction
hinge_pos = env.hinge_position
print("hinge pos: ", hinge_pos)
print("object pos: ", env.obj_pos)
hinge_direction = np.array([0,0,1])

# h_point, f_point, h_direction = hinge_pos, obs['robot0_eef_pos'], hinge_direction
h_point, f_point, h_direction = joint_pose, obs['robot0_eef_pos'], joint_direction
obs = np.concatenate([
                        h_direction, 
                        (f_point-h_point-np.dot(f_point-h_point, h_direction)*h_direction)
                                    ])
done = False
last_grasp_pos = final_grasp_pos
for i in range(200):
    # action = np.zeros_like(env.action_spec[0])
    action = policy.select_action(obs)
    # print("action: ", action)
    action = action * 0.01
    # action[2] = 0
    action = np.concatenate([last_grasp_pos + action,rotation_vector, [1]])
    next_obs, reward, done, _ = env.step(action)
    # h_point, f_point, h_direction = hinge_pos, next_obs['robot0_eef_pos'], hinge_direction
    h_point, f_point, h_direction = joint_pose, next_obs['robot0_eef_pos'], joint_direction
    last_grasp_pos = next_obs['robot0_eef_pos']
    next_obs = np.concatenate([
                                    h_direction, 
                                    (f_point-h_point-np.dot(f_point-h_point, h_direction)*h_direction)
                                ])
    obs = next_obs
    env.render()

env.close()


