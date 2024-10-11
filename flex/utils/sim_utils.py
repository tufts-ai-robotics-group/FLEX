import numpy as np
import open3d as o3d
import robosuite.utils.camera_utils as camera_utils
import os
import copy
from scipy.spatial.transform import Rotation as R 
from robosuite.utils.transform_utils import quat2mat, mat2euler, convert_quat, euler2mat
import copy
from perceptions.interactive_perception import InteractivePerception 
import robosuite as suite
# import mujoco_py

def flip_image(image):
    return np.flip(image, 0)

def get_pointcloud(env, obs, camera_names, 
                   cam_heights, cam_widths, 
                   geom_name:list):
    """
    Create poincloud from depth image and segmentation image, can have multiple cameras
    """
    assert isinstance(camera_names, list) or isinstance(camera_names, str), "camera_names should be a list or a string"
    if isinstance(camera_names, str):
        camera_names = [camera_names]
        cam_heights = [cam_heights]
        cam_widths = [cam_widths]
    # get depth image, rgb image, segmentation image
    # print(obs.keys())
    masked_pcd_list = []

    for camera_name in camera_names:
        cam_width = cam_widths[camera_names.index(camera_name)]
        cam_height = cam_heights[camera_names.index(camera_name)]
        seg_image = obs["{}_segmentation_element".format(camera_name)]
        depth_image = obs['{}_depth'.format(camera_name)]
        depth_image = flip_image(depth_image) 
        # depth_image = np.flip(depth_image, 2)
        depth_image = camera_utils.get_real_depth_map(env.sim, depth_image)
        
        # get geom id corresponding to geom_name
        # pc_geom_id = [env.sim.model.geom_name2id(pc_geo) for pc_geo in env.sim.model.geom_names if 'cube' in pc_geo or 'table' in pc_geo]
        pc_geom_id = [env.sim.model.geom_name2id(pc_geo) for pc_geo in env.sim.model.geom_names if any(name in pc_geo for name in geom_name)] 
        # print('geom name: ', geom_name)
        # print('pc_geom_id:', pc_geom_id)

        # create mask using geom_id and segmentation image
        masked_segmentation = np.isin(seg_image, pc_geom_id)
        masked_segmentation = flip_image(masked_segmentation) 
        # masked_segmentation = np.flip(masked_segmentation, 2)

        # mask the depth image
        masked_depth = np.multiply(depth_image, masked_segmentation).astype(np.float32)

        # create open3d image object
        masked_depth_image = o3d.geometry.Image(masked_depth)

        # get camera intrinsic and extrinsic parameters
        intrinisc_cam_parameters_numpy = camera_utils.get_camera_intrinsic_matrix(env.sim, camera_name, cam_height, cam_width)
        extrinsic_cam_parameters= camera_utils.get_camera_extrinsic_matrix(env.sim, camera_name)

        cx = intrinisc_cam_parameters_numpy[0][2]
        cy = intrinisc_cam_parameters_numpy[1][2]
        fx = intrinisc_cam_parameters_numpy[0][0]
        fy = intrinisc_cam_parameters_numpy[1][1]

        intrinisc_cam_parameters = o3d.camera.PinholeCameraIntrinsic(cam_width, #width 
                                                            cam_height, #height
                                                            fx,
                                                            fy,
                                                            cx,
                                                            cy)
        
        # create open3d pointcloud object
        masked_pcd = o3d.geometry.PointCloud.create_from_depth_image(masked_depth_image,                                                       
                                                intrinisc_cam_parameters
                                                )
        masked_pcd.transform(extrinsic_cam_parameters)

        #estimate normals
        masked_pcd.estimate_normals()
        #orientation normals to camera
        masked_pcd.orient_normals_towards_camera_location(extrinsic_cam_parameters[:3,3])
        masked_pcd_list.append(copy.deepcopy(masked_pcd))
    
    # for i in range(len(masked_pcd_list) - 1):
    #     complete_masked_pcd = masked_pcd_list[i] + masked_pcd_list[i+1]
    complete_masked_pcd = masked_pcd_list[0]
    for i in range(1, len(masked_pcd_list)):
        complete_masked_pcd += masked_pcd_list[i]
    if len(masked_pcd_list) == 1:
        complete_masked_pcd = masked_pcd_list[0]
    return complete_masked_pcd

def initialize_robot_position(env, desired_position):
    obs, reward, done, _ = env.step(np.zeros(env.action_dim))
    # print(obs["gripper_pos"])
    while np.square(obs["gripper_pos"]*50 - desired_position*50).sum() > 0.01:
        action = -(obs["gripper_pos"] - desired_position)
        # normalize the action scale
        action = action / np.linalg.norm(action) * 0.5
        action = np.concatenate([action, np.zeros(4)])
        obs, reward, done, _ = env.step(action)
        # print(obs["gripper_pos"])
        env.render()
    return obs

def open_gripper(env):
    for i in range(5):
        obs, reward, done, _ = env.step(np.concatenate([np.zeros(env.action_dim - 1),[-1]]))
        env.render()
    return obs

def close_gripper(env):
    for i in range(15):
        obs, reward, done, _ = env.step(np.concatenate([np.zeros(env.action_dim - 1),[1]]))
        env.render()
    return obs 

# object utils
def fill_custom_xml_path(xml_path):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    abs_xml_path = os.path.join(base_dir, "sim_models", xml_path)
    return abs_xml_path 

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

def init_camera_pose(env, camera_pos, scale_factor, camera_quat=None):
    '''
    Converts the camera pose to the fixed relative pose of the object in the environment
    '''
    m1 = quat2mat(np.array([-0.5, -0.5, 0.5, 0.5])) # Camera local frame to world frame front, set camera fram
    
    obj_quat = env.obj_quat # wxyz
    obj_quat = convert_quat(obj_quat, to='xyzw')
    rotation_mat_world = quat2mat(obj_quat)
    rotation_euler_world = mat2euler(rotation_mat_world)
    rotation_euler_cam = np.array([rotation_euler_world[2], 0,0])
    
    m3_world = quat2mat(obj_quat)
    m3 = euler2mat(rotation_euler_cam)# Turn camera and microwave simultaneously  
    if camera_quat is not None:
        m2 = quat2mat(camera_quat)  
    else:
        m2 = quat2mat(np.array([-0.005696068282031459, 0.19181093598117868, 0.02913152799094475, 0.9809829120433564])) # Turn camera to microwave

    M = np.dot(m1,m2)
    M = np.dot(M, m3.T) 

    quat = R.from_matrix(M).as_quat() 
    obj_pos = env.obj_pos 
    # obj_pos = np.array([-0.1, -0.5, 0.8])
    camera_pos = np.array(camera_pos)
    camera_trans = scale_factor*camera_pos 
    camera_trans = np.dot(m3_world, camera_trans) 
    res = copy.deepcopy(camera_trans)

    set_camera_pose(env, 'sideview', obj_pos + camera_trans, quat) 
    return res

def get_close_to_grasp_point(env, grasp_pos, grasp_quat, robot_gripper_pos, robot_config, render=False, object_type='microwave'):
    '''
    Get close to the grasp point and quaternion 
    Returns the final grasp position
    ''' 
    # change quat to euler
    sci_rotation = R.from_quat(grasp_quat)
    further_rotation = R.from_euler('z', 90, degrees=True)
    sci_rotation = sci_rotation * further_rotation
    rotation_vector = sci_rotation.as_rotvec()

    mid_point_pos = (grasp_pos + robot_gripper_pos) / 2
    direction = np.array([
        2*(grasp_quat[0]*grasp_quat[2]+grasp_quat[-1]*grasp_quat[1]),
        2*(grasp_quat[1]*grasp_quat[2]-grasp_quat[-1]*grasp_quat[0]),
        1-2*(grasp_quat[0]**2+grasp_quat[1]**2)
    ])
    prepare_offset = robot_config['prepare_offset']
    # prepaer_grasp_pos = grasp_pos + np.array(prepare_offset)
    prepaer_grasp_pos = grasp_pos - direction * prepare_offset[-1]
    if render:
        print('object pos: ', env.obj_pos)
        print('prepare grasp pos: ', prepaer_grasp_pos) 
        print('rotation vector: ', rotation_vector)
        # marker = {
        #     'type': 'box',  # Marker shape
        #     'size': [0.02, 0.02, 0.02],  # Size of the marker (diameter)
        #     'pos': prepaer_grasp_pos,  # Position of the marker
        #     'rgba': [1.0, 0.0, 0.0, 1.0],  # Color and transparency (Red, Opaque)
        #     # 'label': 'Point in World Frame',  # Optional label
        #     # 'label_pos': point_position + np.array([0, 0, 0.1]),  # Label position
        # }
        # # env.sim.model.geom_pos[env.sim.model.geom_name2id("marker_geom")] = prepaer_grasp_pos
        # # env.sim.model.geom_rgba[env.sim.model.geom_name2id("marker_geom")] = marker['rgba']
        # env.sim.add_marker(**marker)
        # env.sim.forward()
        # for _ in range(100):
        #     env.render()


    action = np.concatenate([robot_gripper_pos, rotation_vector, [-1]]) 

    for i in range(120):
        action = np.concatenate([prepaer_grasp_pos, rotation_vector, [-1]])
        obs, rwd, done, _ = env.step(action)
        if render:
            env.render()
            # print(obs['robot0_eef_pos'])

    if render:
        print('reached prepare point')
    final_offset = robot_config['final_offset']
    # final_grasp_pos = grasp_pos + np.array(final_offset)
    final_grasp_pos = grasp_pos - direction * final_offset[-1]
    
    if render:
        print('final grasp pos: ', final_grasp_pos)

    for i in range(80):
        action = np.concatenate([final_grasp_pos, rotation_vector, [-1]])
        obs, rwd, done, _ = env.step(action)
        # env.step(action)
        if render:
            env.render()
            # print(obs['robot0_eef_pos'])

    for i in range(80):
        action = np.concatenate([final_grasp_pos, rotation_vector, [1]])
        obs, rwd, done, _ = env.step(action)
        # obs = env.step(action) 
        if render:
            env.render()
            # print(obs['robot0_eef_pos'])
    
    return final_grasp_pos, rotation_vector, obs

def interact_estimate_params(env, 
                             interaction_kp, 
                             interaction_action_scale, 
                             interaction_timesteps, 
                             rotation_vector, 
                             final_grasp_pos,
                             obs, 
                             directions=['forward', 'backward', 'left', 'right'], 
                             object_type='microwave'):
    '''
    Interacts with the object shortly to estimate the joint type and parameters
    Returns a diction of type and corresponding parameters
    '''
    trajectory = []

    # Update the robot controller to interact mode 
    # old_controller_config = env.robots[0].controller_config 
    env.robots[0].controller_config['kp'] = interaction_kp
    env.robots[0].controller = suite.controllers.controller_factory(env.robots[0].controller_config['type'], env.robots[0].controller_config)

    # Get the robot base position
    robot = env.robots[0]
    base_position = robot.sim.data.get_body_xpos(robot.robot_model.root_body) 
    # last_grasp_pos = obs["robot0_eef_pos"] 
    # print('eef: ', obs)
    last_grasp_pos = final_grasp_pos
    # print(last_grasp_pos)
    # trajectory.append(last_grasp_pos)

    if 'forward' in directions:
        for i in range(interaction_timesteps):
            direction = last_grasp_pos - base_position 
            # print('direction: ', direction)
            direction[2] = 0 # NO Z-AXIS MOVEMENT 
            direction *= interaction_action_scale
            action = np.concatenate([last_grasp_pos + direction, rotation_vector, [1]])
            # action = np.concatenate([obs, [0, 0, 0.1]])
            obs, reward, done, _ = env.step(action)
            trajectory.append(obs["robot0_eef_pos"])
            # env.render() 

    if 'backward' in directions:
        for i in range(interaction_timesteps):
            direction = -(last_grasp_pos - base_position)
            direction[2] = 0 # NO Z-AXIS MOVEMENT
            direction *= interaction_action_scale
            action = np.concatenate([last_grasp_pos + direction, rotation_vector, [1]])
            # action = np.concatenate([obs, [0, 0, 0.1]])
            obs, reward, done, _ = env.step(action)
            trajectory.append(obs["robot0_eef_pos"])

    if 'left' in directions:
        for i in range(interaction_timesteps):
            direction = last_grasp_pos - base_position
            direction[2] = 0 
            z = np.array([0, 0, 1])
            direction = np.cross(z, direction) * interaction_action_scale
            action = np.concatenate([last_grasp_pos + direction, rotation_vector, [1]])
            # action = np.concatenate([obs, [0, 0, 0.1]])
            obs, reward, done, _ = env.step(action)
            trajectory.append(obs["robot0_eef_pos"]) 
    
    if 'right' in directions:
        for i in range(interaction_timesteps):
            direction = last_grasp_pos - base_position
            direction[2] = 0 
            z = np.array([0, 0, 1])
            direction = np.cross(direction, z) * interaction_action_scale
            action = np.concatenate([last_grasp_pos + direction, rotation_vector, [1]])
            # action = np.concatenate([obs, [0, 0, 0.1]])
            obs, reward, done, _ = env.step(action)
            trajectory.append(obs["robot0_eef_pos"]) 
    # print(base_position)
    # Interact
    ip = InteractivePerception(np.array(trajectory)) 

    # prismatic_error, pris_direction = ip.prismatic_error() 
    prismatic_error, pris_direction = ip.prismatic_error_new()
    revolute_error, center, radius, axis = ip.revolute_error()
    print('prismatic error: ', prismatic_error)
    print('revolute error: ', ip.revolute_error()[0])
    print('revolute radius: ', ip.revolute_error()[2]) 
    print('estimated center: ', center)

    # Make sure the estimated prismatic direction is pointing to the robot
    direction =  base_position - trajectory[-1]
    # print('direction: ', direction)
    direction[2] = 0 # NO Z-AXIS MOVEMENT 
    print('direction: ', direction)
    if np.dot(pris_direction, direction) < 0:
        pris_direction = -pris_direction
    print('estimated prismatic direction: ', pris_direction)
    # Make sure the estimated revolute axis supports clockwise rotation
    if np.dot(np.cross(last_grasp_pos-center, direction), axis) < 0:
        axis = -axis

    # Compare the mse error between the prismatic and revolute joint estimation
    # The revolute is considered overfitted if the estimated radius is too large
    print('axis: ', axis)
    if prismatic_error < revolute_error or radius > 3.: 
        estimation_dict = {
            'joint_type': 'prismatic',
            'joint_direction': pris_direction
        }
    else: 
        estimation_dict = {
            'joint_type': 'revolute',
            'joint_position': center,
            'joint_radius': radius,
            'joint_direction': axis
        } 
    # print(np.array(trajectory))
    return obs, estimation_dict, trajectory[-1]
