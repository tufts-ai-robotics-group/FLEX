import numpy as np
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import EmptyArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat

from objects.custom_objects import EvalPrismaticObjects 
from scipy.spatial.transform import Rotation as R
from copy import deepcopy 
import imageio

import os 
# os.environ['MUJOCO_GL'] = 'osmesa'


class RobotPrismaticEnv(SingleArmEnv):
    '''
    Environment for the prismatic object manipulation task.

    Args:
        rotate_around_robot: bool, whether to rotate the object around the robot (when True, x_range and y_range are ignored)
    '''
    def __init__(
        self,
        robots,
        object_name = "trashcan-1",
        object_model_idx = 1,
        scale_object=False, 
        object_scale=1.0,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
        agentview_camera_pos=[0.5986131746834771, -4.392035683362857e-09, 1.5903500240372423], 
        agentview_camera_quat=[0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349], 

        # Object rotation
        obj_rotation=(-np.pi/2, -np.pi / 2),
        x_range = (-1,-1),
        y_range = (0,0),
        z_offset = 0.0,

        rotate_around_robot = False,
        object_robot_distance = (0.5, 0.5),

        move_robot_away=True, 

        # video stuff
        cache_video=False,
        video_width=256,
        video_height=256,
    ):
        
        available_objects = EvalPrismaticObjects.available_objects()
        assert object_name in available_objects, "Invalid object!"

        self.object_name = object_name
        self.placement_initializer = placement_initializer
        self.table_full_size = (0.8, 0.3, 0.05)

        self.object_model_idx = object_model_idx 
        self.obj_rotation = obj_rotation 
        self.x_range = x_range
        self.y_range = y_range
        self.z_offset = z_offset

        self.move_robot_away = move_robot_away

        self.scale_object = scale_object
        self.object_scale = object_scale

        self.rotate_around_robot = rotate_around_robot
        self.object_robot_distance = object_robot_distance 
        self.cache_video=cache_video 
        self.video_width = video_width
        self.video_height = video_height 
        self.frames = []
        self.is_prismatic = True


        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )
        
        self.reward_scale = reward_scale
        self.use_object_obs = use_object_obs
        self.reward_shaping = reward_shaping
        self.placement_initializer = placement_initializer

        
        ## AO-Grasp required information
        self.agentview_camera_pos = agentview_camera_pos 
        self.agentview_camera_quat = agentview_camera_quat 

        self.camera_cfgs = {
            'agentview': {
                'trans': np.array(self.agentview_camera_pos), 
                'quat': np.array(self.agentview_camera_quat)
            }
        }

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # set robot position
        # xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0]) 
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](0)
        # self.robot_init_xpos = xpos
        if self.move_robot_away:
            xpos = (20, xpos[1], 20) # Move the robot away
            self.robots[0].robot_model.set_base_xpos(xpos)
        else:
            xpos = (xpos[0], xpos[1] + 0.3, xpos[2]) # Move the robot away
            self.robots[0].robot_model.set_base_xpos(xpos)

        self.robot_base_xpos = deepcopy(xpos)

        # set empty arena
        mujoco_arena = EmptyArena()

        # set camera pose
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.5986131746834771, -4.392035683362857e-09, 1.5903500240372423],
            quat=[0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349],
        )

        self.prismatic_object = EvalPrismaticObjects(name=self.object_name, 
                                                     scaled=self.scale_object,
                                                     scale=self.object_scale,)

        # get the actual placement of the object
        actual_placement_x, actual_placement_y, actual_placement_rotation, actual_placement_reference_pos = self.get_object_actual_placement()   

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.prismatic_object)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.prismatic_object,
                x_range = actual_placement_x, #No randomization
                y_range = actual_placement_y, #No randomization 
                z_offset=self.z_offset,
                rotation=actual_placement_rotation, #No randomization
                rotation_axis="z",
                ensure_object_boundary_in_range=False, 
                reference_pos=actual_placement_reference_pos
                # reference_pos=(0, 0, 0)
            )

        # Create task
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.prismatic_object)
    
    def get_object_actual_placement(self):
        if self.rotate_around_robot:
            robot_pos = self.robot_base_xpos
            # sample a rotation angle from obj_rotation
            angle = np.random.uniform(self.obj_rotation[0], self.obj_rotation[1])
            # sample a distance from object_robot_distance
            distance = np.random.uniform(self.object_robot_distance[0], self.object_robot_distance[1])

            # calculate correct x and y position using distance and angle
            x = robot_pos[0] + distance * np.cos(angle)
            y = robot_pos[1] + distance * np.sin(angle)

            # we don't use the randomization for x and y
            actual_placement_x = (x, x)
            actual_placement_y = (y, y)
            actual_placement_rotation = (angle, angle)
            actual_placement_reference_pos = (0, 0, 0.5)
        else:
            actual_placement_x = self.x_range
            actual_placement_y = self.y_range
            actual_placement_rotation = self.obj_rotation
            actual_placement_reference_pos = (-0.6, -1.0, 0.5)

        return actual_placement_x, actual_placement_y, actual_placement_rotation, actual_placement_reference_pos

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        self.slider_qpos_addr = self.sim.model.get_joint_qpos_addr(self.prismatic_object.joint) 
        obj_id = self.sim.model.body_name2id(f'{self.prismatic_object.naming_prefix}main')
        self.obj_pos = self.sim.data.body_xpos[obj_id] 
        self.obj_quat = self.sim.data.body_xquat[obj_id]

        self.joint_position_rel = self.prismatic_object.joint_pos_relative
        self.joint_direction_rel = self.prismatic_object.joint_direction / np.linalg.norm(self.prismatic_object.joint_direction)
        self.joint_range = self.prismatic_object.joint_range
        self.joint_position = self.calculate_joint_pos_absolute()
        self.joint_direction = self.calculate_joint_direction_absolute()
        # self.joint_direction_rel = self.prismatic_object.joint_direction / np.linalg.norm(self.prismatic_object.joint_direction)

        self.joint_range = self.prismatic_object.joint_range

    def calculate_joint_pos_absolute(self):
        '''
        calculate the joint position in the world frame
        '''
        joint_pos = np.array(self.joint_position_rel)
    
        joint_pos += deepcopy(self.sim.data.get_body_xpos(self.prismatic_object.prismatic_body))

        return joint_pos
    
    def calculate_joint_direction_absolute(self):
        '''
        calculate the joint direction in the world frame
        '''
        joint_dir = np.array(self.joint_direction_rel)
        joint_dir = np.dot(self.sim.data.get_body_xmat(self.prismatic_object.prismatic_body), joint_dir)
        return joint_dir

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()
        modality = "object"
        pf = self.robots[0].robot_model.naming_prefix
        @sensor(modality=modality)
        def gripper_pos(obs_cache):
            return obs_cache[f"{pf}eef_pos"] if f"{pf}eef_pos" in obs_cache else np.zeros(3)
        @sensor(modality=modality)
        def gripper_quat(obs_cache):
            return obs_cache[f"{pf}eef_quat"] if f"{pf}eef_quat" in obs_cache else np.zeros(3)
        @sensor(modality=modality)
        def handle_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.prismatic_object_handle_site_id])     
        @sensor(modality=modality) 
        def handle_quat(obs_cache):
            return convert_quat(np.array(self.sim.data.body_xquat[self.prismatic_object_handle_site_id]), to="xyzw")
        
        # @sensor(modality=modality)
        # def joint_qpos(obs_cache):
        #     return np.array([self.sim.data.qpos[self.joint_qpos_addr]])

        @sensor(modality=modality)
        def joint_position(obs_cache):
            return self.joint_position
        
        @sensor(modality=modality)
        def joint_direction(obs_cache):
            return self.joint_direction

        # @sensor(modality=modality)
        # def force_point(obs_cache):
        #     return self.relative_force_point_to_world(self.force_point)
        
        # @sensor(modality=modality)
        # def force_point_relative_to_start(obs_cache):
        #     return self.relative_force_point_to_world(self.force_point) - self.initial_force_point

        sensors = [gripper_pos, gripper_quat, joint_position, joint_direction]
        names = [s.__name__ for s in sensors]

        # Create observables
        for name, s in zip(names, sensors):
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
            )

        return observables
    
    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step) 
        if self.cache_video and self.has_offscreen_renderer:
                frame = self.sim.render(self.video_width, self.video_height, camera_name='frontview')
                # print(frame.shape)
                frame = np.flip(frame, 0)
                
                self.frames.append(frame)

    def _reset_internal(self):
        super()._reset_internal()
        self.frames = [] 

        # if rotate_around_robot is True, need to reset the object placement parameters
        if self.rotate_around_robot:
            actual_placement_x, actual_placement_y, actual_placement_rotation, actual_placement_reference_pos = self.get_object_actual_placement()
            self.placement_initializer.x_range = actual_placement_x
            self.placement_initializer.y_range = actual_placement_y
            self.placement_initializer.rotation = actual_placement_rotation
            self.placement_initializer.reference_pos = actual_placement_reference_pos

        object_placements = self.placement_initializer.sample()

        # We know we're only setting a single object (the drawer), so specifically set its pose
        drawer_pos, drawer_quat, _ = object_placements[self.prismatic_object.name]
        drawer_body_id = self.sim.model.body_name2id(self.prismatic_object.root_body)
        self.sim.model.body_pos[drawer_body_id] = drawer_pos
        self.sim.model.body_quat[drawer_body_id] = drawer_quat
        self.handle_current_progress = self.sim.data.qpos[self.slider_qpos_addr] 

    def _check_success(self):
        # TODO: modify this to check if the drawer is fully open 
        joint_qpos = self.sim.data.qpos[self.slider_qpos_addr]

        joint_pos_relative_to_range = (joint_qpos - self.joint_range[0]) / (self.joint_range[1] - self.joint_range[0])
        return joint_pos_relative_to_range > 0.8
    
    def reward(self, action):
        return 0
    
    def _get_camera_config(self, camera_name):
        '''
        Get the information of cameras.
        Input:
            -camera_name: str, the name of the camera to be parsed 

        Returns:
            {
                
        }
        ''' 
        camera_id = self.sim.model.camera_name2id(camera_name)
        camera_pos = self.sim.data.cam_xpos[camera_id] 
        camera_rot_matrix = self.sim.data.cam_xmat[camera_id].reshape(3, 3)

        # Convert the rotation matrix to a quaternion
        camera_quat = R.from_matrix(camera_rot_matrix).as_quat()
        return {
            'camera_config': {
                'trans': camera_pos, 
                'quat': camera_quat
            }
        }

    def set_open_percentage(self, percent):
        '''
        Set the open percent of the drawer
        Input:
            -percent: float, the percent of the drawer to be opened
        '''
        # get the joint range
        joint_range = self.prismatic_object.joint_range

        self.sim.data.qpos[self.slider_qpos_addr] = percent * (joint_range[1] - joint_range[0]) + joint_range[0]
        self.sim.forward()

    @classmethod
    def available_objects(cls):
        available_objects = {
            "prismatic": ["trashcan-1", "canbinet-1"], 
            "cabinet": ["cabinet-1", "cabinet-2", "cabinet-3"],
        }
        return available_objects 
    
    def save_video(self, video_path='videos/robot_prismatic.mp4'):
        imageio.mimsave(video_path, self.frames, fps=120)
        # imageio.imwrite(video_path, self.frames[0])