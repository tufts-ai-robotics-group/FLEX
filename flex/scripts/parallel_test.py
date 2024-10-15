from env.robot_prismatic_env import RobotPrismaticEnv 
from env.robot_revolute_env import RobotRevoluteOpening
import open3d as o3d
import robosuite as suite
from utils.sim_utils import get_pointcloud, flip_image
from utils.gpd_utils import gpd_get_grasp_pose
import numpy as np
import matplotlib.pyplot as plt
from utils.sim_utils import initialize_robot_position, get_close_to_grasp_point, interact_estimate_params
import robosuite.utils.transform_utils as transform
# from utils.control_utils import PathPlanner, Linear, Gaussian
import os 
import json
from scipy.spatial.transform import Rotation as R 
import utils.sim_utils as sim_utils
from grasps.aograsp.get_pointclouds import get_aograsp_ply_and_config
from grasps.aograsp.get_affordance import get_affordance_main
from grasps.aograsp.get_proposals import get_grasp_proposals_main
# from gamma.get_joint_param import get_joint_param_main
from agent.td3 import TD3
from termcolor import colored, cprint
import termcolor
import warnings
import yaml 
# from perceptions.interactive_perception import InteractivePerception 
from env.wrappers import make_env, SubprocessEnv, GymWrapper
# from env.train_multiple_revolute_env import MultipleRevoluteEnv
# from env.train_prismatic_env import TrainPrismaticEnv
warnings.filterwarnings("ignore", category=FutureWarning) 
import stable_baselines3 as sb3
import argparse
import time

revolute_state = lambda state: np.concatenate([
    state['hinge_direction'], 
    state['force_point']-state['hinge_position']-np.dot(state['force_point']-state['hinge_position'], state['hinge_direction'])*state['hinge_direction']
    ]) 

prismatic_state = lambda obs: np.concatenate([obs["joint_direction"], obs["force_point_relative_to_start"]]) 


def get_state(obs, params, final_grasp_pos):
    '''
    Get the state from observation and estimated parameters
    '''
    if params['joint_type'] == 'prismatic':
        direction = params['joint_direction'] 
        progress = obs['robot0_eef_pos'] - final_grasp_pos
        return np.concatenate([direction, progress])
    else:
        hinge_direction = params['joint_direction'] 
        hinge_position = params['joint_position']
        force_point = obs['robot0_eef_pos'] 
        projection = force_point - hinge_position - np.dot(force_point - hinge_position, hinge_direction) * hinge_direction 
        return np.concatenate([hinge_direction, projection])
        # return revolute_state(obs)

def test_object(run_id, obj_name='cabinet-1', object_scale=0.4, robot_name='Panda', use_sb3_agent=False, save_video=False, render_env=False):

    controller_name = "OSC_POSE"
    controller_cfg_dir = os.path.join('cfg',"controller_configs")
    controller_cfg_path = os.path.join(controller_cfg_dir, "osc_pose_absolute.json")
    controller_configs = suite.load_controller_config(custom_fpath = controller_cfg_path,default_controller=controller_name)

    with open(controller_cfg_path, 'r') as f:
        controller_configs = json.load(f)

    with open('cfg/test_config.yaml') as f:
        config = yaml.safe_load(f) 

    experiment_name = config['experiment_name'] 
    test_objects = config['test_objects'] 
    # run_id = config['run_id']

    force_policy_config = config['force_policy'] 
    environment_config = config['environment'] 
    robot_config = config['robot'] 

    # RL environment configs
    max_timesteps = environment_config['max_timesteps'] 
    action_repeat = environment_config['action_repeat']

    # Robot configs
    action_scale = robot_config['action_scale'] 
    interaction_action_scale = robot_config['interaction_action_scale']
    kp_interaction = robot_config['kp_interaction']
    kp_execution = robot_config['kp_execution']
    interaction_timesteps = robot_config['interaction_timesteps'] 
    interaction_directions = robot_config['interaction_directions'] 
    # render = True

    x_range = (0.5, 0.5)
    y_range = (0.5, 0.5)

    # x_range = (1, 1)
    # y_range = (1, 1)
    env_kwargs = dict(
        robots=robot_name,
        object_name = obj_name,
        obj_rotation=(-np.pi/2, -np.pi/2),
        rotate_around_robot=False, 
        scale_object = True,
        object_scale = object_scale,
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
        camera_widths = [1024,1024,1024,1024],
        move_robot_away = False,
        x_range = x_range,
        y_range = y_range, 
        cache_video = save_video, 
        video_width = 512,
        video_height = 512,
        object_robot_distance=(0.7, 0.7)
    )
    # env_name = "RobotPrismaticEnv"
    env_name = 'RobotRevoluteOpening' if env_kwargs['object_name'].split('-')[0] in ['microwave', 'dishwasher', 'train'] else 'RobotPrismaticEnv'

    env = suite.make(
        env_name,
        **env_kwargs
    )

    obs = env.reset()

    env.set_open_percentage(0.40) 
    env.sim.forward() 
    if use_sb3_agent:
        prismatic_agent = sb3.TD3.load(f'checkpoints/force_policies/{experiment_name}/prismatic/{run_id}/prismatic_final')
        revolute_agent = sb3.TD3.load(f'checkpoints/force_policies/{experiment_name}/revolute/{run_id}/revolute_final')
    else:
        prismatic_agent = TD3(0.001, 6, 3, 5)
        prismatic_agent.load(f'checkpoints/force_policies/{experiment_name}/prismatic/{run_id}', 'prismatic_final')
        revolute_agent = TD3(0.001, 6, 3, 5)
        revolute_agent.load(f'checkpoints/force_policies/{experiment_name}/revolute/{run_id}', 'revolute_final')
    # revolute_agent = sb3.TD3.load(f'checkpoints/force_policies/{experiment_name}/revolute/{run_id}/force_td3_revolute_1000000')

    robot_gripper_pos = obs["robot0_eef_pos"]

    # get the joint qpos
    reset_joint_qpos = env.sim.data.qpos[env.slider_qpos_addr]

    # sim_utils.init_camera_pose(env, camera_pos=np.array([-0.77542536, -0.02539806,  0.30146208]), scale_factor=3) 

    # if want to visualize the point cloud in open3d, the environment later will not work
    viz_imgs = False
    need_o3d_viz = False
    run_cgn = True
    pcd_wf_path = f'point_clouds/world_frame_pointclouds/world_frame_{env_kwargs["object_name"]}.ply'
    pcd_wf_no_downsample_path = f'point_clouds/world_frame_pointclouds/world_frame_{env_kwargs["object_name"]}_no_downsample.ply'
    camera_info_path = f'infos/camera_info_{env_kwargs["object_name"]}.npz'


    camera_euler = np.array([ 30., 22., 3.])
    camera_euler_for_pos = np.array([camera_euler[-1], camera_euler[1], -camera_euler[0]])
    camera_rotation = R.from_euler("xyz", camera_euler_for_pos, degrees=True)
    camera_quat = R.from_euler("xyz", camera_euler, degrees=True).as_quat()
    camera_forward = np.array([1, 0, 0])
    world_forward = camera_rotation.apply(camera_forward)
    distance = 1.5
    camera_pos = -distance * world_forward
    print('old env obj pos: ', env.obj_pos)

    x_range = (1, 1)
    y_range = (1, 1)

    pcd_wf_path, pcd_wf_no_downsample_path,camera_info_path = get_aograsp_ply_and_config(o_env=env,env_name = env_name, 
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
                            reset_x_range = x_range, 
                            reset_y_range = y_range,
                            )

    # get the affordance heatmap
    pcd_cf_path, affordance_path = get_affordance_main(pcd_wf_path, camera_info_path, 
                                                    viz=False)


    # get the grasp proposals
    world_frame_proposal_path, top_k_pos_wf, top_k_quat_wf = get_grasp_proposals_main(
        pcd_cf_path, 
        affordance_path, 
        camera_info_path, 
        run_cgn=run_cgn, 
        viz=need_o3d_viz, 
        save_wf_pointcloud=True,
        object_name=env_kwargs["object_name"],
        top_k=10,
    )

    object_pos = np.array(env.obj_pos)

    top_k_pos_wf = top_k_pos_wf + object_pos

    grasp_pos = top_k_pos_wf[0]
    grasp_quat = top_k_quat_wf[0] 

    final_grasp_pos, rotation_vector, obs = get_close_to_grasp_point(env, grasp_pos, grasp_quat, robot_gripper_pos, 
                                                                     robot_config=config[robot_name], render=render_env)

    done = False
    last_grasp_pos = final_grasp_pos
    obs, estimation_dict, final_grasp_pos = interact_estimate_params(
        env=env, 
        interaction_kp=kp_interaction, 
        interaction_action_scale=interaction_action_scale, 
        interaction_timesteps=interaction_timesteps, 
        rotation_vector=rotation_vector, 
        final_grasp_pos=final_grasp_pos, 
        obs=obs, 
        directions=interaction_directions
        )

    state = get_state(obs, estimation_dict, final_grasp_pos) 
    # print('ip estimated joint direction: ', estimation_dict['joint_direction']) 
    # print('gt joint direction: ', obs['joint_direction'])

    if estimation_dict['joint_type'] == 'prismatic' and env.is_prismatic: 
        classification_correct = True
        print('joint classfication correct') 
    elif estimation_dict['joint_type'] == 'revolute' and not env.is_prismatic:
        classification_correct = True
        print('joint classfication correct') 
    else:
        classification_correct = False
        # env.save_video(f'videos/ip_failure_video_{run_id}.mp4') 
        # env.close()
        # return 0, 0
        # print('joint classfication incorrect')

    if estimation_dict['joint_type'] == 'prismatic':
        policy = prismatic_agent
    else:
        policy = revolute_agent

    for i in range(max_timesteps):
        if use_sb3_agent:
            action, _ = policy.predict(state)
        else:
            action = policy.select_action(state)
        # print(action)
        action = action.flatten()
        # action = np.array([0, 5, 0])
        action[-1] = 0
        action += np.array([1, 1, 0])
        action = action * action_scale 
        # print('state: ', state)
        # print('action: ', action)
        # print(action)
        # action = -action if estimation_dict['joint_type'] == 'revolute' else action
        # if obj_name.split('-')[0] == 'microwave':
        #     action = -action
        action = np.concatenate([last_grasp_pos + action, rotation_vector, [1]]) 
        next_obs, reward, done, _ = env.step(action)
        if done or env._check_success():
            break
        last_grasp_pos = next_obs['robot0_eef_pos']
        state = get_state(next_obs, estimation_dict, final_grasp_pos)
        if render_env:
            # print('rendering')
            env.render()

    # print('success' if env._check_success() else 'failed')
    success = env._check_success()

    # if not success:
        # print('prismatic error: ', ip.prismatic_error())
    # print('revolute error: ', ip.revolute_error()[0])
    # print('revolute radius: ', ip.revolute_error()[2]) env.save_video(f'videos/failure_test_video_{run_id}.mp4')
    # if save_video:
    #     env.save_video(f'videos/{robot_name}_{obj_name}.mp4')
    # print('gt joint direction: ', obs['joint_direction'])
    env.close()
    return classification_correct, success

def test(args):
    with open ('cfg/test_config.yaml') as f:
        config = yaml.safe_load(f)
    objects = config['test_objects'] 
    render = args.render
    print(render)
    print('render env: ', type(render))
    train_objects = config['train_objects']
    save_video = False
    robots = config['robots']
    if args.run_id is not None:
        policy_run_id = args.run_id
    else:
        policy_run_id = config['run_id']
    n_trials = config['n_trials']
    object_config = config['object_params']
    evaluation_start = time.time()
    
    for robot in robots:
        logdir = f'results/novel_object_experiment/run_{policy_run_id}/{robot}'
        os.makedirs(logdir, exist_ok=True) 
        json_dict = {}
        for object in objects:
            print(object)
            classification_results, execution_results = [], []
            obj_type = object.split('-')[0]
            if obj_type =='train':
                obj_type = object.split('-')[1]
            for i in range(n_trials):
                # print(robot)
                classification_correct, success = test_object(run_id=policy_run_id, obj_name=object, object_scale=object_config[obj_type]['scale'], robot_name=robot, use_sb3_agent=False, save_video=save_video, render_env=render)
                classification_results.append(classification_correct)
                execution_results.append(success)
            if obj_type not in json_dict:
                json_dict[obj_type] = {}
            json_dict[obj_type][object] = {
                'execution_success_rate': np.mean(execution_results), 
                'classification_success_rate': np.mean(classification_results)
            }
        with open(f'{logdir}/results.json', 'w') as f:
            json.dump(json_dict, f)
    evaluation_end = time.time()
    print(f'Evaluation time: {evaluation_end - evaluation_start}')
    
    # for robot in robots:
    #     logdir = f'results/seen_object_experiments/run_{policy_run_id}/{robot}'
    #     os.makedirs(logdir, exist_ok=True)
    #     json_dict = {}
    #     for object in train_objects:
    #         classification_results, execution_results = [], []
    #         for i in range(n_trials):
    #             classification_correct, success = test_object(run_id=policy_run_id, obj_name=object, robot_name=robot)
    #             classification_results.append(classification_correct)
    #             execution_results.append(success)
    #         obj_type = object.split('-')[0]
    #         if obj_type not in json_dict:
    #             json_dict[obj_type] = {}
    #         json_dict[obj_type][object] = {
    #             'execution_success_rate': np.mean(execution_results), 
    #             'classification_success_rate': np.mean(classification_results)
    #         }
    #     with open(f'{logdir}/results.json', 'w') as f:
    #         json.dump(json_dict, f)


if __name__ == '__main__':

    # objects = [
    #     'cabinet-1',
    #     'cabinet-2',
    #     'cabinet-3', 
    #     'trashcan-1',
    #     'trashcan-2',
    #     'trashcan-3',
    #     'microwave-1', 
    #     'microwave-2',
    #     'microwave-3',
    #     'microwave-4',
    #     'microwave-5',
    #     'microwave-6',
    #     'dishwasher-1',
    #     'dishwasher-2',

    # ]
    # json_dict = {}
    # run_id = 7
    # for object in objects:
    #     ss = []
    #     for i in range(20):
    #         classification_correct, success = test_object(i)
    #         ss.append(success)
    #     object_type = object.split('-')[0]
    #     if object_type not in json_dict:
    #         json_dict[object_type] = {}
    #     json_dict[object_type][object] = np.mean(ss)
    # with open(f'results_{run_id}.json', 'w') as f:
    #     json.dump(json_dict, f)
    # print(np.mean(ss))
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=int)
    parser.add_argument('--render', type=bool,default=False)
    args = parser.parse_args()
    test(args)