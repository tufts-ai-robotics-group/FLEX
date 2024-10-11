"""
Run pointscore inference on partial pointcloud
"""

import os
import pickle
from argparse import ArgumentParser
import numpy as np
import torch
import open3d as o3d

# from train_model.test import load_conf
import aograsp.viz_utils as v_utils
import aograsp.model_utils as m_utils 
import utils.aograsp_utils.dataset_utils as d_utils 
import utils.aograsp_utils.rotation_utils as r_utils
import copy 
import termcolor



def inference_affordance(model, pcd_wf_path, camera_info_path, device): 
    '''
    Converts a world frame point cloud to camera frame and saves it under "point_clouds/camera_frame_pointclouds/camera_frame_$(object_name).ply", then
    inference the affordance of a point cloud using the AO-Grasp model. The camera frame affordance is saved in "outputs/point_score/camera_frame_$(object_name)_affordance.npz" and the heatmap image is saved in "outputs/point_score_img/heatmap_$(object_name).png"
    model: AO-Grasp model loaded with m_utils. 

    pcd_wf_path: path to the point cloud. Must be the world frame. Currently starts with "point_clouds/world_frame_pointclouds/world_frame_$(object_name).ply" 
    camera_info_path: path to the information of camera of the pcd. Currently starts with "infos/$(object_name)_camera_info.npz 

    saves the mean and the unnormalized point clouds in camera frame to "point_clouds/camera_frame_pointclouds/camera_frame_$(object_name).ply"
    '''
    # Read from args.pcd_path and put into input_dict for model
    if pcd_wf_path is None:
        raise ValueError("Missing path to input point cloud (--pcd_path)")

    pcd_ext = os.path.splitext(pcd_wf_path)[-1]
    if pcd_ext == ".ply":
        # Open3d pointcloud
        pcd = o3d.io.read_point_cloud(pcd_wf_path)
        pcd = pcd.farthest_point_down_sample(4096) 
        ########################################################
        # Convert loaded point cloud into c frame
        pts_arr = np.array(pcd.points) 
        print('pts_arr mean: ', np.mean(pts_arr, axis=0))
        pts_arr = d_utils.get_aograsp_pts_in_cam_frame_z_front_with_info(pts_arr, camera_info_path) 
    else:
        raise ValueError(f"{pcd_ext} filetype not supported")

    # Recenter pcd to origin 
    # pcd = pcd.farthest_point_down_sample(4096)
    mean = np.mean(pts_arr, axis=0) 
    print('pts cf mean: ', mean)
    pts_arr_backup = copy.deepcopy(pts_arr)

    pts_arr -= mean 

    # Randomly shuffle points in pts
    np.random.shuffle(pts_arr)

    # Get pts as tensor and create input dict for model
    pts = torch.from_numpy(pts_arr).float().to(device) 
    pts = torch.unsqueeze(pts, dim=0)
    input_dict = {"pcs": pts} 
    # print(pts) 

    # Run inference
    print("-------------------Computing heatmap-------------------")
    model.eval()
    with torch.no_grad():
        test_dict = model.test(input_dict, None)

    # Save heatmap point cloud 
    print(termcolor.colored(f'inference result: {test_dict.keys()}', 'yellow'))
    scores = test_dict["point_score_heatmap"][0].cpu().numpy() 
    heatmap_dict = {
        "pts": pts_arr,  # Save original un-centered data
        "labels": scores,
    }

    # Save the converted points and the mean into a npz.
    data_name = os.path.splitext(os.path.basename(pcd_wf_path))[0].replace('world_frame_', '')

    cf_points = o3d.geometry.PointCloud() 
    cf_points.points = o3d.utility.Vector3dVector(pts_arr_backup)

    # save camera frame point cloud and heatmap
    pcd_cf_path = 'point_clouds/camera_frame_pointclouds/'+'camera_frame_'+data_name+'.ply'

    o3d.io.write_point_cloud(pcd_cf_path, cf_points) 
    print(termcolor.colored(f'Saved camera frame point cloud to: {pcd_cf_path}', "blue")) 

    affordance_path = 'outputs/point_score/'+'camera_frame_'+data_name+'_affordance.npz'

    np.savez_compressed(affordance_path, data=heatmap_dict)
    print(termcolor.colored(f'Saved heatmap to {affordance_path}', 'blue'))
    return test_dict, pts_arr, pcd_cf_path, affordance_path

def get_affordance_main(pcd_wf_path, camera_info_path, device="cuda:0", viz=False):
    '''
    main function for getting affordance
    '''

    # load ao-grasp model
    model = m_utils.load_model(
        model_conf_path='checkpoints/grasp_models/aograsp_models/conf.pth', 
        ckpt_path='checkpoints/grasp_models/aograsp_models/770-network.pth'
    )
    model.to(device)

    # do the inference
    t_dict, pts, pcd_cf_path, affordance_path = inference_affordance(model, pcd_wf_path, camera_info_path, device) 
    if viz:
        visualize_heatmap(t_dict, pts, point_score_dir = None, point_score_img_dir = None, data_name = "1")
    return pcd_cf_path, affordance_path



def visualize_heatmap(test_dict, pts, point_score_dir, point_score_img_dir, data_name):
    scores = test_dict["point_score_heatmap"][0].cpu().numpy()
    # pcd_path = os.path.join(point_score_dir, f"{data_name}.npz")

    pts_arr = pts
    mean = np.mean(pts_arr, axis=0) 
    pts_arr -= mean
    heatmap_dict = {
        "pts": pts_arr,  # Save original un-centered data
        "labels": scores,
    }

    # Save image of heatmap
    if point_score_img_dir:
        fig_path = os.path.join(point_score_img_dir, f"heatmap_{data_name}.png")
    else:
        fig_path = None
    if point_score_img_dir:
        hist_path = os.path.join(point_score_img_dir, f"heatmap_{data_name}_hist.png") 
    else: 
        fig_path = None
    try:
        v_utils.viz_heatmap(
            heatmap_dict["pts"],
            scores,
            save_path=fig_path,
            frame="camera",
            scale_cmap_to_heatmap_range=True,
        )
    except Exception as e:
        print(e)

    v_utils.viz_histogram(
        scores,
        save_path=hist_path,
        scale_cmap_to_heatmap_range=True,
    )
    # print(f"Heatmap saved to: {pcd_path}")
    print(termcolor.colored(f"Visualization saved to: {fig_path}", 'blue'))


def parse_args():
    parser = ArgumentParser()

    # main parameters (required)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Path to directory to write output files",
    )
    parser.add_argument("--pcd_wf_path", type=str, help="Path to seg_pcd_clean.ply") 

    parser.add_argument('--camera_info_path', type=str, help='Path to information.npz') 

    parser.add_argument('--object', type=str, default='temp_door')

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="cpu or cuda:x for using cuda on GPU x",
    )

    parser.add_argument(
        "--display",
        type=str,
        choices=[":1", ":2", ":3"],
        help="Display number; for accomdating remote desktop setups",
    )

    # parse args
    conf = parser.parse_args()

    conf.data_path = None

    return conf



if __name__ == "__main__":
    args = parse_args() 
    model = m_utils.load_model(
        model_conf_path='checkpoints/grasp_models/aograsp_models/conf.pth', 
        ckpt_path='checkpoints/grasp_models/aograsp_models/770-network.pth'
    )
    model.to(args.device)
    pcd_wf_path = args.pcd_wf_path 
    camera_info_path = args.camera_info_path
    device = args.device
    t_dict, pts, _, _ = inference_affordance(model, pcd_wf_path, camera_info_path, device) 
    # pts = o3d.io.read_point_cloud(pcd_path) 
    visualize_heatmap(t_dict, pts, '/home/wgao22/projects/ForceRL/outputs/point_score', '/home/wgao22/projects/ForceRL/outputs/point_score_img', 'microwave')
