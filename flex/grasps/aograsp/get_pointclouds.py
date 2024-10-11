'''
Generate the point cloud and camera config from a single camera in the system that AO-grasp supports. 
'''

import torch 
import argparse
import os
import pickle

import robosuite as suite
import open3d as o3d 
import numpy as np
import aograsp.model_utils as m_utils 
import aograsp.viz_utils as v_utils 
import aograsp.data_utils as d_utils 
import aograsp.rotation_utils as r_utils 
import matplotlib.pyplot as plt

from utils.sim_utils import get_pointcloud 
from env.robot_revolute_env import RobotRevoluteOpening 
from scipy.spatial.transform import Rotation as R 
from robosuite.utils.transform_utils import * 
import utils.sim_utils as s_utils
import time
import robosuite.utils.camera_utils as robo_cam_utils
import robosuite.utils.transform_utils as transform_utils 
import termcolor
import imageio

def set_camera_pose(env, camera_name, position, quaternion):
    sim = env.sim
    cam_id = sim.model.camera_name2id(camera_name)
    sim.model.cam_pos[cam_id] = position
    sim.model.cam_quat[cam_id] = quaternion
    sim.forward() 

def display_camera_pose(env, camera_name):
    cam_id = env.sim.model.camera_name2id(camera_name)
    print(f'{camera_name} pose: {env.sim.model.cam_pos[cam_id]}') 
    print(f'{camera_name} quat: {env.sim.model.cam_quat[cam_id]}') 
    
def get_aograsp_ply_and_config(o_env, env_name, env_kwargs: dict,object_name, camera_pos, camera_quat = None,
                               reset_x_range = (1,1), 
                               reset_y_range = (1,1),
                               reset_joint_qpos = np.pi / 6,
                               scale_factor=3, camera_info_path='infos/temp_door_camera_info.npz', 
                               pcd_wf_path='point_clouds/world_frame_pointclouds/world_frame_temp_door.npz', 
                               pcd_wf_no_downsample_path='point_clouds/world_frame_pointclouds/world_frame_temp_door_no_downsample.npz',
                               device='cuda:0', denoise=True, 
                               viz = True , 
                               need_o3d_viz = False,):
    '''
    Get the segmented object point cloud from the environment in world frame and stores it in `pcd_wf_path`. This is in the format of "point_clouds/world_frame_pointclouds/world_frame_$(object_name).ply".
    Stores the camera information into `pcd_wf_path`. This is in the format of "infos/$(object_name)_camera_info.npz".
    '''


    # set move_robot_away to true
    env_kwargs["move_robot_away"] = True

    # disable rotate_around_robot
    # env_kwargs["rotate_around_robot"] = False
    
    # use given x_range and y_range so that we can see the whole object. 
    env_kwargs["x_range"] = reset_x_range
    env_kwargs["y_range"] = reset_y_range
    env_kwargs["camera_names"] = ["sideview"]
    env_kwargs["camera_heights"] = [256]
    env_kwargs["camera_widths"] = [256]
    # env_kwargs["obj_rotation"] = (0,0) 
    env_kwargs['has_renderer'] = True
    env_kwargs['has_offscreen_renderer'] = True
    env_kwargs['use_camera_obs'] = True 
    env_kwargs['camera_depths'] = True
    # viz=True 


    # obj_euler = R.from_quat(o_env.obj_quat).as_euler('xyz', degrees=True)
    # obj_euler_rad = 
    # env_kwargs['obj_rotation'] = (obj_euler[0], obj_euler[0])
    env = suite.make(
        env_name,
        **env_kwargs
    )

    obs = env.reset() 

    env.sim.data.qpos[env.slider_qpos_addr] = reset_joint_qpos
    env.sim.forward()

    cam_pos_actual = s_utils.init_camera_pose(env, camera_pos, scale_factor, camera_quat=camera_quat)
    display_camera_pose(env, 'sideview') 

    low, high = env.action_spec
    obs, reward, done, _ = env.step(np.zeros_like(low))
    # env.render()
    print('new env obj position: ', env.obj_pos)
    
    if viz:
        # plt.imshow(obs['frontview_image'])
        # plt.show()
        rgb = obs['sideview_image']
        rgb = np.flip(rgb, axis=0)
        plt.imshow(rgb) 
        plt.show() 
        image = rgb
        imageio.imwrite('temp.png', image)
        dep = obs['sideview_depth'] 
        dep = np.flip(dep, axis=0)
        plt.imshow(dep)
        plt.show() 

    
    
    pointcloud = get_pointcloud(env, obs, ['sideview'], [256], [256], [object_name]) 
    
    if denoise:
        pointcloud, ind = pointcloud.remove_statistical_outlier(nb_neighbors=200, std_ratio=2.0)
    
    obs_arr_without_downsample = np.asarray(pointcloud.points) - np.array(env.obj_pos)
    obs_arr_without_downsample = obs_arr_without_downsample / scale_factor
    pointcloud_without_downsample = o3d.geometry.PointCloud()
    pointcloud_without_downsample.points = o3d.utility.Vector3dVector(obs_arr_without_downsample)
    o3d.io.write_point_cloud(pcd_wf_no_downsample_path, pointcloud_without_downsample)


    pointcloud = pointcloud.farthest_point_down_sample(4096) 
    obs_arr = np.asarray(pointcloud.points) - np.array(env.obj_pos)
    obs_arr = obs_arr / scale_factor

    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(obs_arr)
    pts_arr = np.array(pointcloud.points)

    o3d.io.write_point_cloud(pcd_wf_path, pointcloud) 
    print(termcolor.colored(f'World frame point cloud saved to {pcd_wf_path}', 'blue')) 

    extrinsic = robo_cam_utils.get_camera_extrinsic_matrix(env.sim, 'sideview')
    obj_euler = R.from_quat(env.obj_quat).as_euler('xyz', degrees=True)
    camera_rot_90 = transform_utils.euler2mat(np.array([0,0,-np.pi-obj_euler[0]])) @ transform_utils.quat2mat(camera_quat) 
    # camera_rot_90 = transform_utils.euler2mat(np.array([0,0,env_kwargs['obj_rotation'][0]])) @ transform_utils.quat2mat(camera_quat) 
    camera_quat_rot_90 = transform_utils.mat2quat(camera_rot_90)
    
    camera_euler = R.from_quat(camera_quat).as_euler('xyz', degrees=True)
    camera_euler_for_gamma = np.array([camera_euler[-1], camera_euler[1], -camera_euler[0]]) 
    camera_rot_for_gamma = R.from_euler('xyz', camera_euler_for_gamma, degrees=True).as_matrix()
    camera_rot_for_gamma = transform_utils.euler2mat(np.array([0,0,env_kwargs['obj_rotation'][0]])) @ camera_rot_for_gamma
    camera_quat_for_gamma = R.from_matrix(camera_rot_for_gamma).as_quat()
    

    camera_config = {'data': {
                        'camera_config': {
                            # 'trans': camera_pos*scale_factor, 
                            'trans': camera_pos, 
                            'quat': camera_quat_rot_90,
                            'trans_absolute': cam_pos_actual,
                            'quat_for_gamma': camera_quat_for_gamma,
                        }
                    }
                } 
    with open(camera_info_path, 'wb') as f:
        pickle.dump(camera_config, f) 

    print(termcolor.colored(f'world frame camera information saved to {camera_info_path}', 'blue'))
    # env.close()
    
    # need to close the environment if want to visualize the point cloud in open3d
    if need_o3d_viz:
        env.close()
    

    return pcd_wf_path, pcd_wf_no_downsample_path, camera_info_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--environment_name', type=str, default='RobotRevoluteOpening') 
    parser.add_argument('--object_name', type=str, default='drawer')
    parser.add_argument('--camera_pos', default=[-0.77542536, -0.02539806,  0.30146208])
    parser.add_argument('--camera_quat', default=[-0.005696068282031459, 0.19181093598117868, 0.02913152799094475, 0.9809829120433564])
    # parser.add_argument('--output_dir', type=str, default='outputs/')
    parser.add_argument('--scale_factor', default=3)
    parser.add_argument('--pcd_wf_path', type=str, default='point_clouds/world_frame_pointclouds/world_frame_temp_door.ply') 
    parser.add_argument('--camera_info_path', type=str, default='temp_door_camera.npz')
    parser.add_argument('--device', type=str, default='cuda:0') 
    parser.add_argument('--denoise', default=True)
    args = parser.parse_args()

    # ENV ARGS
    controller_name = "OSC_POSE"
    controller_configs = suite.load_controller_config(default_controller=controller_name)
    env_kwargs = dict(
    robots="Panda",
    object_type = "microwave",
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
    x_range = (-1,-1),
    y_range = (0,0),
    )
    env_name = "RobotRevoluteOpening"
    get_aograsp_ply_and_config(env_name, env_kwargs, args.object_name, args.camera_pos, 
                           args.camera_quat, args.scale_factor, args.camera_info_path, args.pcd_wf_path, denoise=args.denoise)

