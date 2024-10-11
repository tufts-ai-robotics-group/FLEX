# NOTE: CGN is implemented with tensorflow. Use `cgn` conda environment to run all the utils.

import os
from argparse import ArgumentParser
import numpy as np
# import torch
import open3d as o3d

# from train_model.test import load_conf
# import aograsp.viz_utils as v_utils
# import aograsp.model_utils as m_utils 
import utils.aograsp_utils.dataset_utils as d_utils 
import utils.aograsp_utils.rotation_utils as r_utils 
from scipy.spatial.transform import Rotation
# from robosuite.utils.transform_utils import * 
import matplotlib 
import utils.mesh_utils as mesh_utils


def scale_to_0_1(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data)) 

def get_o3d_pts(pts, normals=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd

def get_eef_line_set_for_o3d_viz(eef_pos_list, eef_quat_list, highlight_top_k=None):
    # Get base gripper points
    g_opening = 0.07
    gripper = mesh_utils.create_gripper("panda")
    gripper_control_points = gripper.get_control_point_tensor(
        1, False, convex_hull=False
    ).squeeze()
    mid_point = 0.5 * (gripper_control_points[1, :] + gripper_control_points[2, :])
    grasp_line_plot = np.array(
        [
            np.zeros((3,)),
            mid_point,
            gripper_control_points[1],
            gripper_control_points[3],
            gripper_control_points[1],
            gripper_control_points[2],
            gripper_control_points[4],
        ]
    )
    gripper_control_points_base = grasp_line_plot.copy()
    gripper_control_points_base[2:, 0] = np.sign(grasp_line_plot[2:, 0]) * g_opening / 2
    # Need to rotate base points, our gripper frame is different
    # ContactGraspNet
    r = Rotation.from_euler("z", 90, degrees=True)
    gripper_control_points_base = r.apply(gripper_control_points_base)

    # Compute gripper viz pts based on eef_pos and eef_quat
    line_set_list = []
    for i in range(len(eef_pos_list)):
        eef_pos = eef_pos_list[i]
        eef_quat = eef_quat_list[i]

        gripper_control_points = gripper_control_points_base.copy()
        g = np.zeros((4, 4))
        rot = Rotation.from_quat(eef_quat).as_matrix()
        g[:3, :3] = rot
        g[:3, 3] = eef_pos.T
        g[3, 3] = 1
        z = gripper_control_points[-1, -1]
        gripper_control_points[:, -1] -= z
        gripper_control_points[[1], -1] -= 0.02
        pts = np.matmul(gripper_control_points, g[:3, :3].T)
        pts += np.expand_dims(g[:3, 3], 0)

        lines = [[0, 1], [2, 3], [1, 4], [1, 5], [5, 6]]
        if highlight_top_k is not None:
            if i < highlight_top_k:
                # Draw grasp in green
                colors = [[0,1,0] for i in range(len(lines))]
            else:
                colors = [[0,0,0] for i in range(len(lines))]
        else:
            colors = [[0,0,0] for i in range(len(lines))]

        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pts),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        line_set_list.append(line_set)

    return line_set_list

def viz_pts_and_eef_o3d(
    pts_pcd,
    eef_pos_list,
    eef_quat_list,
    heatmap_labels=None,
    save_path=None,
    frame="world",
    draw_frame=False,
    cam_frame_x_front=False,
    highlight_top_k=None,
    pcd_rgb=None
):
    """
    Plot eef in o3d visualization, with point cloud, at positions and
    orientations specified in eef_pos_list and eef_quat_list
    pts_pcd, eef_pos_list, and eef_quat_list need to be in same frame
    """
    print('o3ding...')
    pcd = get_o3d_pts(pts_pcd)
    if pcd_rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(pcd_rgb)
    else:
        if heatmap_labels is not None:
            # Scale heatmap for visualization
            heatmap_labels = scale_to_0_1(heatmap_labels)

            cmap = matplotlib.cm.get_cmap("RdYlGn")
            colors = cmap(np.squeeze(heatmap_labels))[:, :3]
            pcd.colors = o3d.utility.Vector3dVector(colors)

    # Get line_set for drawing eef in o3d
    line_set_list = get_eef_line_set_for_o3d_viz(
        eef_pos_list, eef_quat_list, highlight_top_k=highlight_top_k,
    )

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.add_geometry(pcd)

    for line_set in line_set_list:
        vis.add_geometry(line_set)

    # Draw ref frame
    if draw_frame:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )
        vis.add_geometry(mesh_frame)

    # Move camera
    if frame == "camera":
        # If visualizing in camera frame, view pcd from scene view
        ctr = vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        fov = ctr.get_field_of_view()
        H = np.eye(4)
        if cam_frame_x_front:
            R = Rotation.from_euler("XYZ", [90, 0, 90], degrees=True).as_matrix()
            H[:3, :3] = R
        param.extrinsic = H
        ctr.convert_from_pinhole_camera_parameters(param)
    else:
        print('world frame')
        # If world frame, place camera accordingly to face object front
        ctr = vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        fov = ctr.get_field_of_view()
        H = np.eye(4)
        H[2, -1] = 1
        R = Rotation.from_euler("XYZ", [90, 0, 90], degrees=True).as_matrix()
        H[:3, :3] = R
        param.extrinsic = H
        ctr.convert_from_pinhole_camera_parameters(param)

    vis.poll_events()
    vis.update_renderer()

    if save_path is None:
        vis.run()
    else:
        vis.capture_screen_image(
            save_path,
            do_render=True,
        )
    vis.destroy_window()