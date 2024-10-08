# AO-Grasp scripts

`get_pointclouds.py` segments the object from the robosuite environment, generates a partial point cloud from `sideview` camera in world frame, and stores the camera information in `infos/$(object)_camera_info.npz`. 
The point cloud is stored in `point_clouds/world_frame_pointclouds/world_frame_$(object).ply`. 

`get_affordance.py` takes the world frame point cloud and the camera information as input, stores the z-front camera frame point cloud in `point_clouds/camera_frame_pointclouds/camera_frame_$(object).ply`, and stores the heatmap 
in `outputs/point_score/camera_frame_$(object)_affordance.npz`. 

`get_cf_affordance.py` uses `cgn` conda environment. It takes the camera frame heatmap and the camera information as input, generates the grasps proposed be cgn in `outputs/grasp_proposals/camera_frame_proposals/camera_frame_$(object)_grasp.npz`. 