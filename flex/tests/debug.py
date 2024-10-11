import numpy as np 
import open3d as o3d 
from scipy.spatial.transform import Rotation as R

cf_points = o3d.io.read_point_cloud('point_clouds/camera_frame_pointclouds/camera_frame_cabinet-1.ply')
cf_arr = np.asarray(cf_points.points)
print('camera frame mean: ', np.mean(cf_arr, axis=0))

wf_points = o3d.io.read_point_cloud('point_clouds/world_frame_pointclouds/world_frame_cabinet-1.ply')
wf_arr = np.asarray(wf_points.points)
print('world frame mean: ', np.mean(wf_arr, axis=0)) 

camera_info = info_dict = np.load('infos/camera_info_cabinet-1.npz', allow_pickle=True)['data']['camera_config']
# print('camera info: ', camera_info) 
print('camera pos: ', camera_info['trans'])
print('camera quat: ', camera_info['quat']) 
# print('gamma quat: ', camera_info['quat_for_gamma'])
rotation = R.from_quat(camera_info['quat']) 

calculated_wf_points = np.dot(rotation.as_matrix(), cf_arr.T).T + camera_info['trans'] 
print('calculated world frame mean: ', np.mean(calculated_wf_points, axis=0))