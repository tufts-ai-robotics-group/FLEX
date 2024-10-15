# new force-based training environment with multiple revolute type objects
# environment with the door model from original robosuite code
# Issue: the self.last_force_point_xpos is not correctly reset, however in normal steps it is corret.
from copy import deepcopy

import numpy as np
import robosuite as suite
from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
import robosuite.macros as macros
from robosuite.models.arenas import EmptyArena
from objects.custom_objects import TrainRevoluteObjects
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.transform_utils import convert_quat


from robosuite.environments.base import MujocoEnv
import mujoco

from robosuite.utils import OpenCVRenderer
from robosuite.utils.binding_utils import MjRenderContextOffscreen
from utils.renderer_modified import MjRendererForceVisualization
import imageio

class MultipleRevoluteEnv(MujocoEnv):
    def __init__(self, 
                 object_name = "door_original",
                 init_object_angle=(-np.pi / 2.0 - 0.25, 0),
                 use_camera_obs=False, 
                 placement_initializer=None,
                 random_force_point = False,
                 has_renderer=True, 
                 has_offscreen_renderer=False,
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
                 camera_segmentations=None,
                 renderer="mujoco",
                 action_scale=1,
                 reward_scale=10, 
                 progress_scale=1e3, 
                 renderer_config=None, 
                 debug_mode=False, 

                 # Save video stuff 
                 save_video=False,
                 video_width=256,
                 video_height=256,
                 ): 
        '''
        Door environment with new parameters for curriculum learning.

        New parameters:
            - init_object_angle: tuple, the initial rotation angle of the door. Increase the range as the curricula proceeds.
        '''
        
        self.action_scale = action_scale
        self.has_renderer = has_renderer
        self.has_offscreen_renderer = has_offscreen_renderer
        self.use_camera_obs = use_camera_obs
        self.placement_initializer = placement_initializer
        self.debug_mode = debug_mode 
        self.init_object_angle = init_object_angle 
        self.progress_scale = progress_scale

        self.frames = []
        self.video_width = video_width 
        self.video_height = video_height 
        self.cache_video = save_video

        
        self.object_name = object_name
        assert self.object_name in ["train-door-counterclock-1", "door_original", "train-microwave-1", "train-dishwasher-1"], "Invalid object name"
        # save bounds for relative force point for each object, (lower bound, upper bound, axis)
        self.train_object_force_point_bound = {
            "train-door-counterclock-1" : [(0,0), (0, 0.3), (-0.7, 0.7)],
            "train-microwave-1" : [(0, 0), (-0.35, 0.35), (0., 0.5)], 
            "train-dishwasher-1" : [(0, 0), (-0.55, 0.), (-0.6, 0.6)], 
            "door_original" : [(-0.22,0), (0, 0.), (-0.29,0.29)]
        }

        self.use_object_obs = True # always use low-level object obs for this environment
        self.random_force_point = random_force_point
        self.is_prismatic = False
        super().__init__(
            has_renderer=self.has_renderer,
            has_offscreen_renderer=self.has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            renderer=renderer,
            renderer_config=renderer_config
        )

        self.horizon = horizon 
        self._action_dim = 3
        self.reward_scale = reward_scale
        

        # Set camera attributes
        self.camera_names =  (
            list(camera_names) if type(camera_names) is list or type(camera_names) is tuple else [camera_names]
        )
        self.num_cameras = len(self.camera_names)
        self.camera_heights = self._input2list(camera_heights, self.num_cameras)
        self.camera_widths = self._input2list(camera_widths, self.num_cameras)
        self.camera_depths = self._input2list(camera_depths, self.num_cameras)
        self.camera_segmentations = self._input2list(camera_segmentations, self.num_cameras)

        seg_is_nested = False
        for i, camera_s in enumerate(self.camera_segmentations):
            if isinstance(camera_s, list) or isinstance(camera_s, tuple):
                seg_is_nested = True
                break
        camera_segs = deepcopy(self.camera_segmentations)
        for i, camera_s in enumerate(self.camera_segmentations):
            if camera_s is not None:
                self.camera_segmentations[i] = self._input2list(camera_s, 1) if seg_is_nested else deepcopy(camera_segs)
        if self.use_camera_obs and not self.has_offscreen_renderer:
            raise ValueError("Error: Camera observations require an offscreen renderer!")
        if self.use_camera_obs and self.camera_names is None:
            raise ValueError("Must specify at least one camera name when using camera obs")
        
        # Set up backup renderer
        self.backup_renderer = mujoco.Renderer(self.sim.model._model)

    def visualize(self, vis_settings):
        return super().visualize(vis_settings=vis_settings)

    @property
    def _visualizations(self):
        """
        Visualization keywords for this environment

        Returns:
            set: All components that can be individually visualized for this environment
        """
        vis_settings = super()._visualizations
        return vis_settings
    
    @property
    def action_spec(self):
        '''
        bounds for the action space
        '''
        # TODO: check if this is correct
        low, high = np.concatenate(np.ones(3),np.zeros(3)).astype(float) , np.concatenate(np.ones(3),np.zeros(3)).astype(float)
        return low, high
    
    @property
    def action_dim(self):
        """
        Size of the action space

        Returns:
            int: Action space dimension
        """
        return self._action_dim
    
    @staticmethod
    def _input2list(inp, length):
        """
        Helper function that converts an input that is either a single value or a list into a list

        Args:
            inp (None or str or list): Input value to be converted to list
            length (int): Length of list to broadcast input to

        Returns:
            list: input @inp converted into a list of length @length
        """
        # convert to list if necessary
        return list(inp) if type(inp) is list or type(inp) is tuple else [inp for _ in range(length)]
    
    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # set empty arena
        mujoco_arena = EmptyArena()

        # Modify default agentview camera
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.5986131746834771, -4.392035683362857e-09, 1.5903500240372423],
            quat=[0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349],
        )

        # initialize objects of interest
        self.revolute_object = TrainRevoluteObjects(
            name=self.object_name,
        )
        # write xml to file
        # with open("train-door-counterclock-1-temp.xml", "w") as f:
            # import xml.etree.ElementTree as ET
            # tree = ET.ElementTree(self.revolute_object.get_obj().root)
            # tree.write(f)

        # import xml.etree.ElementTree as ET
        # root = ET.Element("root")
        # root.append(self.revolute_object.get_obj())
        # tree = ET.ElementTree(root)
        # tree.write("train-dishwasher-temp.xml", encoding="utf-8")

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.revolute_object)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.revolute_object,
                x_range=[-1.07, -1.09],
                y_range=[-0.01, 0.01],
                # rotation=(-np.pi / 2.0 - 0.25, -np.pi / 2.0),
                rotation=self.init_object_angle,

                # rotation=(-np.pi , np.pi),
                # rotation = (-np.pi, -np.pi),
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=(-0.2, -0.35, 0.8), # hard coded as original code
            )

        # Create task
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[],
            mujoco_objects=[self.revolute_object])
        
    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """

        super()._setup_references()
    
        # Additional object references from this env
        self.object_body_ids = dict()
        self.object_body_ids["revolute"] = self.sim.model.body_name2id(self.revolute_object.revolute_body)

        self.hinge_qpos_addr = self.sim.model.get_joint_qpos_addr(self.revolute_object.hinge_joint)
        
        self.hinge_position_rel = self.revolute_object.hinge_pos_relative
        # hinge direction is normalized

        # panel geom
        self.panel_geom_name = self.revolute_object.naming_prefix + "panel"

        self.hinge_direction_rel = self.revolute_object.hinge_direction / np.linalg.norm(self.revolute_object.hinge_direction)

        # TODO: the revolute body pos is not updated when _setup_references. Currently it is calculated in pre_action. Are there 
        # more elegant way to do this?
        self.hinge_position = self.calculate_hinge_pos_absolute()
        self.hinge_direction = self.calculate_hinge_direction_absolute()

        self.hinge_range = self.revolute_object.hinge_range

    def calculate_hinge_pos_absolute(self):
        '''
        calculate the hinge position in the world frame
        '''
        hinge_pos = np.array(self.hinge_position_rel)
        # hinge_pos[2] -= 0.5
    
        # hinge_pos = np.dot(self.sim.data.get_body_xmat(self.revolute_object.revolute_body), hinge_pos)
        # print("body xpos for hinge: ", self.sim.data.get_body_xpos(self.revolute_object.revolute_body))
        hinge_pos += deepcopy(self.sim.data.get_body_xpos(self.revolute_object.revolute_body))

        return hinge_pos
    
    def calculate_hinge_direction_absolute(self):
        '''
        calculate the hinge direction in the world frame
        '''
        hinge_dir = np.array(self.hinge_direction_rel)
        hinge_dir = np.dot(self.sim.data.get_body_xmat(self.revolute_object.revolute_body), hinge_dir)
        return hinge_dir

    def _create_camera_sensors(self, cam_name, cam_w, cam_h, cam_d, cam_segs, modality="image"):
            convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]

            sensors = []
            names = []
            rgb_sensor_name = f"{cam_name}_image"
            depth_sensor_name = f"{cam_name}_depth"
            segmentation_sensor_name = f"{cam_name}_segmentation"

            @sensor(modality=modality)
            def camera_rgb(obs_cache):
                img = self.sim.render(
                    camera_name=cam_name,
                    width=cam_w,
                    height=cam_h,
                    depth=cam_d,
                )
                if cam_d:
                    rgb, depth = img
                    obs_cache[depth_sensor_name] = np.expand_dims(depth[::convention], axis=-1)
                    return rgb[::convention]
                else:
                    return img[::convention]
                
            sensors.append(camera_rgb)
            names.append(rgb_sensor_name)

            if cam_d:

                @sensor(modality=modality)
                def camera_depth(obs_cache):
                    return obs_cache[depth_sensor_name] if depth_sensor_name in obs_cache else np.zeros((cam_h, cam_w, 1))

                sensors.append(camera_depth)
                names.append(depth_sensor_name)
            if cam_segs is not None:
                raise NotImplementedError("Segmentation sensors are not supported for this environment")
            
            return sensors, names
    
    def _create_segementation_sensor(self, cam_name, cam_w, cam_h, cam_s, seg_name_root, modality="image"):
        pass

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            modality = "object"

            # Define sensor callbacks

            @sensor(modality=modality)
            def hinge_qpos(obs_cache):
                return np.array([self.sim.data.qpos[self.hinge_qpos_addr]])

            @sensor(modality=modality)
            def hinge_position(obs_cache):
                return self.hinge_position
            
            @sensor(modality=modality)
            def hinge_direction(obs_cache):
                return self.hinge_direction

            @sensor(modality=modality)
            def force_point(obs_cache):
                return self.relative_force_point_to_world(self.force_point)


            sensors = [hinge_qpos, hinge_position, force_point, hinge_direction]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def sample_relative_force_point(self):
        '''
        sample a point on the door to apply force, the point is relative to the door frame
        '''
        if not self.random_force_point:

            # TODO: for now we only have fixed force point for door_original
            if self.object_name == "door_original":
                return np.array([-0.2, 0, 0])
            else:
                raise NotImplementedError("For now we only have fixed force point for door_original")
        
        else:
            # sample a point on the door panel according to the object type
            panel_size = self.train_object_force_point_bound[self.object_name]
            x = np.random.uniform(panel_size[0][0], panel_size[0][1])
            y = np.random.uniform(panel_size[1][0], panel_size[1][1])
            z = np.random.uniform(panel_size[2][0], panel_size[2][1])
            return np.array([x,y,z])
    
    def relative_force_point_to_world(self, relative_force_point):
        '''
        convert the relative force point to world frame
        '''
        panel_pos = self.sim.data.get_geom_xpos(self.panel_geom_name)
        panel_rot = self.sim.data.get_geom_xmat(self.panel_geom_name)
        
        return deepcopy(panel_pos + np.dot(panel_rot, relative_force_point))

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        # create visualization screen or renderer
        if self.has_renderer and self.viewer is None:
            self.viewer = OpenCVRenderer(self.sim)

            # Set the camera angle for viewing
            if self.render_camera is not None:
                camera_id = self.sim.model.camera_name2id(self.render_camera)
                self.viewer.set_camera(camera_id)

        if self.has_offscreen_renderer:
            if self.sim._render_context_offscreen is None:
                render_context = MjRendererForceVisualization(self.sim, modify_fn=self.modify_scene,device_id=self.render_gpu_device_id)
                # render_context = MjRenderContextOffscreen(self.sim, device_id=self.render_gpu_device_id)
            self.sim._render_context_offscreen.vopt.geomgroup[0] = 1 if self.render_collision_mesh else 0
            self.sim._render_context_offscreen.vopt.geomgroup[1] = 1 if self.render_visual_mesh else 0

        self.frames = []

        # disable gravity
        self.sim.model._model.opt.gravity[:] = [0, 0, 0]
    
        # additional housekeeping
        self.sim_state_initial = self.sim.get_state()
        self._setup_references()
        self.cur_time = 0
        self.timestep = 0
        self.done = False
        self.first_success = True

        # Empty observation cache and reset all observables
        self._obs_cache = {}
        for observable in self._observables.values():
            observable.reset()

        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # We know we're only setting a single object (the door), so specifically set its pose
            door_pos, door_quat, _ = object_placements[self.revolute_object.name]
            revolute_body_id = self.sim.model.body_name2id(self.revolute_object.root_body)
            self.sim.model.body_pos[revolute_body_id] = door_pos
            self.sim.model.body_quat[revolute_body_id] = door_quat
        
        

        # set the door hinge to the initial position, lower bound of the hinge limit
        self.sim.data.qpos[self.hinge_qpos_addr] = self.hinge_range[0]

        # get hinge absolute position
        self.hinge_position = self.calculate_hinge_pos_absolute()
        self.hinge_direction = self.calculate_hinge_direction_absolute()

        # read the body position and rotation as the relative zero
        self.revolute_body_rel_zero_pos = deepcopy(self.sim.data.get_body_xpos(self.revolute_object.revolute_body))
        self.revolute_body_rel_zero_rot = deepcopy(self.sim.data.get_body_xmat(self.revolute_object.revolute_body))
        
        # sample a force point on the door
        self.force_point = self.sample_relative_force_point()
        self.force_point_world = deepcopy(self.relative_force_point_to_world(self.force_point))
        
        # get the initial render arrow end
        self.render_arrow_end = self.force_point_world

        

        # get the initial hinge qpos
        self.last_hinge_qpos = self.sim.data.qpos[self.hinge_qpos_addr]
        self.last_force_point_xpos = deepcopy(self.force_point_world)
        # print("initial force point: ", self.force_point_world)

    def step(self, action):
        if self.cache_video and self.has_offscreen_renderer:
            frame = self.sim.render(self.video_width, self.video_height, camera_name='sideview')
            self.frames.append(frame) 
        return super().step(action)

    def _pre_action(self, action, policy_step=False):
        self.hinge_position = self.calculate_hinge_pos_absolute()
        self.hinge_direction = self.calculate_hinge_direction_absolute()

        # if self.cache_video and self.has_offscreen_renderer:
        #         frame = self.sim.render(self.video_width, self.video_height, camera_name='sideview')
        #         self.frames.append(frame)
        
        self.last_hinge_qpos = deepcopy(self.sim.data.qpos[self.hinge_qpos_addr])
        # get the arrow end position
        self.force_point_world = self.relative_force_point_to_world(self.force_point)
        self.render_arrow_end = self.force_point_world + action * self.action_scale

        assert len(action) == self.action_dim, "environment got invalid action dimension -- expected {}, got {}".format(
            self.action_dim, len(action))

        # manually set qfrc_applied to 0 to avoid the force applied in the last step
        self.sim.data._data.qfrc_applied = [0]

        # apply the force
        mujoco.mj_applyFT(self.sim.model._model, self.sim.data._data, 
                          action * self.action_scale, np.zeros(3), self.force_point_world, 
                          self.object_body_ids["revolute"], self.sim.data._data.qfrc_applied)

    def _check_success(self):
        # TODO:implement this
        hinge_qpos = self.sim.data.qpos[self.hinge_qpos_addr]

        hinge_pos_relative_to_range = (hinge_qpos - self.hinge_range[0]) / (self.hinge_range[1] - self.hinge_range[0])
        # open 30 degrees
        return hinge_pos_relative_to_range > 0.8
    
    def reward(self, action = None):
        if self._check_success() and self.first_success == True:
            # print("success")
            self.first_success = False
            return 5
        elif self._check_success() and self.first_success == False:
            return 1
        elif self.first_success == False and not self._check_success():
            return -10
        else:
            current_force_point_xpos = deepcopy(self.relative_force_point_to_world(self.force_point))
            # print("current force point: ", current_force_point_xpos)
            # print("relative hinge position: ", self.hinge_position_rel)
            # print("last force point: ", self.last_force_point_xpos)
            delta_force_point = current_force_point_xpos - self.last_force_point_xpos
            self.last_force_point_xpos = deepcopy(current_force_point_xpos)
            
            # print("delta force point: ", delta_force_point)
            delta_force_point *= 1000
            valid_force_reward = np.dot(action, delta_force_point) / (np.linalg.norm(action) + 1e-7)
            valid_force_reward = np.abs(valid_force_reward)
            
            progress = self.sim.data.qpos[self.hinge_qpos_addr] - self.last_hinge_qpos
            
            # print("progress: ", progress)
            valid_force_reward = valid_force_reward * np.sign(progress)

            if np.abs(progress) < 7e-6:
                valid_force_reward = -0.1
                progress = -0.01
                self.current_step_force_projection = 0
            else:
                self.current_step_force_projection = np.dot(action, delta_force_point) / (np.linalg.norm(action) * np.linalg.norm(delta_force_point))

            if self.debug_mode:
                print(f'valid force reward: {valid_force_reward}') 
                print(f'progress reward: {progress}')
                # print the angle between the delta force point and the action
                print("angle between action and delta force point: ", np.arccos(np.dot(action, delta_force_point) / (np.linalg.norm(action) * np.linalg.norm(delta_force_point))) * 180 / np.pi)
            
            if np.isnan(valid_force_reward):
                print("nan reward")
            # print("valid force reward: ", valid_force_reward)

            return valid_force_reward * self.reward_scale + self.progress_scale * progress
            # return valid_force_reward * self.reward_scale
    
    @property
    def current_action_projection(self):
        return self.current_step_force_projection

    
    def modify_scene(self, scene):
        rgba = np.array([0.5, 0.5, 0.5, 1.0])
        
        # point1 = body_xpos
        point1 = self.force_point_world
        point2 = self.render_arrow_end
        # point1 = self.hinge_position
        # point2 = self.hinge_position + self.hinge_direction 

        radius = 0.05

        if scene.ngeom >= scene.maxgeom:
            return
        scene.ngeom += 1

        mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                      mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3),
                      np.zeros(3), np.zeros(9), rgba.astype(np.float32))
        mujoco.mjv_connector(scene.geoms[scene.ngeom-1],
                           int(mujoco.mjtGeom.mjGEOM_ARROW), radius,
                           point1, point2)

    def render(self):
        super().render() 

    def save_video(self, path): 
        if not self.cache_video:
            print("No video to save")
            return
        imageio.mimsave(path, self.frames, fps=30)

    def step(self, action):
        next_state, reward, done, info = super().step(action) 
        if self._check_success(): 
            # print('success')
            done = True
        return next_state, reward, done, info
    
    @property
    def success(self):
        return self._check_success()

    @classmethod
    def available_objects(cls):
        available_objects = {
            "single-obj": ["door_original"],
            "door-like": ["door_original", "train-door-counterclock-1", "train-microwave-1"],
            "dishwasher-like": ["train-dishwasher-1"],
            "all": ["door_original", "train-door-counterclock-1", "train-microwave-1", "train-dishwasher-1"]
        }
        return available_objects