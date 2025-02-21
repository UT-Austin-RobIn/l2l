from collections import OrderedDict

import cv2
import numpy as np
import xml.etree.ElementTree as ET

from gymnasium import spaces

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import MultiTableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.utils.transform_utils import convert_quat
import robosuite.utils.transform_utils as T
from robosuite.utils.camera_utils import CameraMover

import torch
from l2l.envs.robosuite.oracle_base import OracleBase

class WalledMultiStage(SingleArmEnv):

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.4, 0.4, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
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
        pi_format='discrete',
        text_encoder=None,
        partitioned_encoding=False,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array([(0.0, 0, 0.8), (0, -0.6, 0.8), (0, 0.6, 0.8)])

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        self.stage_id = 0

        #################  language encoding related  #################
        self.pi_format = pi_format
        self.text_encoder = text_encoder
        self.partitioned_encoding = partitioned_encoding

        self.pi_to_text = {}
        self.pi_to_text['pick_blue'] = [
                                        # "Lift up the blue cude",
                                        "Pick up the blue cube",
                                        # "Lift the blue cube",
                                        # "Grasp the blue cube"
                                    ]
        self.pi_to_text['pick_wood'] = [
                                        # "Lift up the wooden cude",
                                        # "Pick up the wooden cube",
                                        "Lift the wooden cube",
                                        # "Grasp the wooden cube"
                                    ]
        self.pi_to_text['place_on_red'] = [
                                        "Place the cube on the red region",
                                        # "Put the cube on the red region",
                                        # "Place the cube on the red area",
                                        # "Put the cube on the red area"
                                    ]
        self.pi_to_text['place_on_green'] = [
                                        # "Place the cube on the green region",
                                        # "Put the cube on the green region",
                                        # "Place the cube on the green area",
                                        "Put the cube on the green area"
                                    ]
        self.current_text_embeds = None

        ### End of language encoding related ###

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

        cam_tree = ET.Element("camera", attrib={"name": "activeview"})
        self.CAMERA_NAME = cam_tree.get("name")
        self.setting_camera = False
        self.initial_camera_quat = [0.405, 0.405, 0.581, 0.581, ]  # xyzw
        self.n_camera_actions = 9
        self.total_env_steps_taken = 0

        self.recorded_obs = []

    @property
    def action_space(self):
        low, high = self.action_spec
        return spaces.Box(low, high)

    def reward(self, action):
        return 0

    def staged_rewards(self):
        return 0, 0, 0

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = list(self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0]))
        xpos[0] += self.table_offset[0][0]
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = MultiTableArena(
            table_full_sizes=self.table_full_size,
            table_frictions=self.table_friction,
            table_offsets=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        mujoco_arena.set_camera(
            camera_name="activeview",
            pos=[0.6, 0, 1.45],
            # pos=[0.6, np.random.uniform(-0.5, 0.5), 1.45 + np.random.uniform(-0.5, 0.5)], #[0.6, 0, 1.45],
            quat=[0.653, 0.271, 0.271, 0.653],#[0.581, 0.405, 0.405, 0.581]
        )

        # mujoco_arena.set_camera(
        #     camera_name="agentview",
        #     pos=[self.table_offset[0][0]+0.5, 0.0, 1.35],
        #     quat=[0.653, 0.271, 0.271, 0.653]
        # )

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        bluewood = CustomMaterial(
            texture="WoodBlue",
            tex_name="bluewood",
            mat_name="handle1_mat",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
        )
        
        darkwood = CustomMaterial(
            texture="WoodDark",
            tex_name="darkwood",
            mat_name="MatDarkWood",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "3 3", "specular": "0.4","shininess": "0.1"}
        )

        whitewood = CustomMaterial(
            texture="BricksWhite",
            tex_name="brickwhite",
            mat_name="MatWhiteBrick",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "3 3", "specular": "0.4","shininess": "0.1"}
        )

        self.wall = BoxObject(
            name="wall",
            size_min=[0.02, 0.13, 0.15],
            size_max=[0.02, 0.13, 0.15],
            rgba=[0, 0, 0, 1],
            material=whitewood,
        )

        self.cube_blue = BoxObject(
            name="cube_blue",
            size_min=[0.02, 0.02, 0.02],
            size_max=[0.02, 0.02, 0.02],
            rgba=[0, 0, 0, 1],
            material=bluewood,
        )

        self.cube_wood = BoxObject(
            name="cube_wood",
            size_min=[0.02, 0.02, 0.02],
            size_max=[0.02, 0.02, 0.02],
            rgba=[0, 0, 0, 1],
            material=darkwood,
        )

        self.serving_region_red = BoxObject(
            name="serving_region_red",
            size_min=[0.05, 0.05, 0.001],
            size_max=[0.05, 0.05, 0.001],
            rgba=[1, 0, 0, 1],
            material=redwood,
        )

        self.serving_region_green = BoxObject(
            name="serving_region_green",
            size_min=[0.05, 0.05, 0.001],
            size_max=[0.05, 0.05, 0.001],
            rgba=[0, 1, 0, 1],
            material=greenwood,
        )
        
        self.pick_blue = np.random.rand() >= 0.5
        self.cube_reference_pick = BoxObject(
            name="cube_reference_pick",
            size_min=[0.04, 0.04, 0.04],
            size_max=[0.04, 0.04, 0.04],
            rgba=(0, 0, 0, 1),
            material=bluewood if self.pick_blue else darkwood,
        )
        
        self.place_on_red = np.random.rand() >= 0.5
        self.cube_reference_place = BoxObject(
            name="cube_reference_place",
            size_min=[0.05, 0.05, 0.05],
            size_max=[0.05, 0.05, 0.05],
            rgba=(0, 0, 0, 1),
            material=redwood if self.place_on_red else greenwood,
        )

        self.placement_initializer = self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        
        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="ObjectSampler-wall",
                mujoco_objects=self.wall,
                x_range=[0.1, 0.1],
                y_range=[0, 0],
                rotation=[0, 0, 0],
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset[1],
                z_offset=0.01,
        ))

        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="ObjectSampler-cube_blue",
                mujoco_objects=self.cube_blue,
                x_range=[0.0, 0.0],
                y_range=[0.1, 0.1],#[-0.15, 0.15],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset[0],
                z_offset=0.01,
        ))

        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="ObjectSampler-cube_wood",
                mujoco_objects=self.cube_wood,
                x_range=[0.0, 0.0],
                y_range=[-0.1, -0.1],#[-0.15, 0.15],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset[0],
                z_offset=0.01,
        ))

        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="ObjectSampler-serving_region_red",
                mujoco_objects=self.serving_region_red,
                x_range=[0.1, 0.1],
                y_range=[0.1, 0.1],#[-0.15, 0.15],
                rotation=(0., 0.),
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset[0],
                z_offset=0.01,
        ))

        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="ObjectSampler-serving_region_green",
                mujoco_objects=self.serving_region_green,
                x_range=[0.1, 0.1],
                y_range=[-0.1, -0.1],#[-0.15, 0.15],
                rotation=(0., 0.),
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset[0],
                z_offset=0.01,
        ))
        
        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="ObjectSampler-cube_reference_pick",
                mujoco_objects=self.cube_reference_pick,
                x_range=[-0.1, -0.1],
                y_range=[-0.1, -0.1],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset[1],
                z_offset=0.01,
        ))

        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="ObjectSampler-cube_reference_place",
                mujoco_objects=self.cube_reference_place,
                x_range=[-0.1, 0.1],
                y_range=[-0.1, 0.1],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset[2],
                z_offset=0.01,
        ))


        mujoco_objects = [self.cube_blue, self.cube_wood, self.serving_region_red, self.serving_region_green, self.cube_reference_pick, self.cube_reference_place, self.wall]
        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=mujoco_objects,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.cube_blue_body_id = self.sim.model.body_name2id(self.cube_blue.root_body)
        self.cube_wood_body_id = self.sim.model.body_name2id(self.cube_wood.root_body)
        self.serving_region_red_body_id = self.sim.model.body_name2id(self.serving_region_red.root_body)
        self.serving_region_green_body_id = self.sim.model.body_name2id(self.serving_region_green.root_body)
        self.cube_reference_pick_body_id = self.sim.model.body_name2id(self.cube_reference_pick.root_body)
        self.cube_reference_place_body_id = self.sim.model.body_name2id(self.cube_reference_place.root_body)

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
    
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
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # position and rotation of the first cube
            @sensor(modality=modality)
            def cube_blue_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube_blue_body_id])

            @sensor(modality=modality)
            def cube_blue_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cube_blue_body_id]), to="xyzw")
            
            @sensor(modality=modality)
            def cube_wood_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube_wood_body_id])
            
            @sensor(modality=modality)
            def cube_wood_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cube_wood_body_id]), to="xyzw")
            
            @sensor(modality=modality)
            def serving_region_red_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.serving_region_red_body_id])
            
            @sensor(modality=modality)
            def serving_region_green_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.serving_region_green_body_id])

            @sensor(modality=modality)
            def gripper_to_cube_blue(obs_cache):
                return (
                    obs_cache["cube_blue_pos"] - obs_cache[f"{pf}eef_pos"]
                    if "cube_blue_pos" in obs_cache and f"{pf}eef_pos" in obs_cache
                    else np.zeros(3)
                )
            
            @sensor(modality=modality)
            def gripper_to_cube_wood(obs_cache):
                return (
                    obs_cache["cube_wood_pos"] - obs_cache[f"{pf}eef_pos"]
                    if "cube_wood_pos" in obs_cache and f"{pf}eef_pos" in obs_cache
                    else np.zeros(3)
                )
            
            @sensor(modality=modality)
            def privileged_info(obs_cache):

                if self.pi_format == 'discrete':
                    info = np.zeros(4).astype(np.int8)
                    # Test factored info as a first step towards continuous privileged info
                    if self.pick_blue:
                        info[0] = 1
                    else:
                        info[1] = 1
                    if self.place_on_red:
                        info[2] = 1
                    else:
                        info[3] = 1

                elif self.pi_format == "onehot":
                    info = np.zeros(4).astype(np.int8)
                    if self.pick_blue:
                        if self.place_on_red:
                            info[0] = 1
                        else:
                            info[1] = 1
                    else:
                        if self.place_on_red:
                            info[2] = 1
                        else:
                            info[3] = 1

                elif self.pi_format == "continuous":
                    info = np.zeros(4).astype(np.float32)
                    if self.pick_blue:
                        if self.place_on_red:
                            info[0] = 1.0
                        else:
                            info[1] = 1.0
                    else:
                        if self.place_on_red:
                            info[2] = 1.0
                        else:
                            info[3] = 1.0

                elif self.pi_format == "clip":
                    info = self.current_text_embeds

                return info

            @sensor(modality=modality)
            def gt_privileged_info(obs_cache):
                return np.array([self.pick_blue, self.place_on_red])
            
            @sensor(modality="extra")
            def stage_id(obs_cache):
                cube = self.cube_blue if self.pick_blue else self.cube_wood

                cube_pos = self.sim.data.body_xpos[self.cube_blue_body_id] if self.pick_blue else self.sim.data.body_xpos[self.cube_wood_body_id]
                cube_height = cube_pos[2]
                table_height = self.model.mujoco_arena.table_offsets[0][2]

                if ((not self._check_grasp(self.robots[0].gripper, cube.contact_geoms)) or (cube_height < table_height+0.03)) \
                    and (self.stage_id == 0):
                    return np.array([1, 0])
                else:
                    self.stage_id = 1
                    return np.array([0, 1])
                
            @sensor(modality='extra')
            def task_obs(obs_cache):
                return np.zeros(3)
            
            @sensor(modality='extra')
            def camera_angle(obs_cache):
                return np.zeros((2,))
            
            @sensor(modality='extra')
            def camera_pos(obs_cache):
                return np.zeros((3,))
            
            @sensor(modality="extra")
            def is_policy_uncertain(obs_cache):
                return np.zeros((1,))

            sensors = [
                cube_blue_pos, cube_blue_quat, cube_wood_pos, cube_wood_quat, serving_region_red_pos,
                serving_region_green_pos, gripper_to_cube_blue, gripper_to_cube_wood, privileged_info,
                stage_id, task_obs, camera_angle, is_policy_uncertain]
            sensors += [gt_privileged_info, camera_pos]
            
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def generate_pi_text(self):
        """
        Generate the text input for the text encoder based on the privileged info
        """

        if self.pick_blue:
            s1 = np.random.choice(self.pi_to_text['pick_blue'])
        else:
            s1 = np.random.choice(self.pi_to_text['pick_wood'])

        if self.place_on_red:
            s2 = np.random.choice(self.pi_to_text['place_on_red'])
        else:
            s2 = np.random.choice(self.pi_to_text['place_on_green'])

        if self.partitioned_encoding:
            return [s1, s2]
        else:
            return [s1 + " and " + s2]

    @torch.no_grad()
    def get_text_embeddings(self, text):
        self.pi_text = text
        inputs = self.text_encoder[0](text, padding=True, return_tensors="pt")
        outputs = self.text_encoder[1](**inputs)
        text_embeds = outputs.text_embeds
        text_embeds = torch.flatten(text_embeds)
        return text_embeds
    
    def _check_success(self):
        """
        Check if blocks are stacked correctly.

        Returns:
            bool: True if blocks are correctly stacked
        """
        cube_pos = self.sim.data.body_xpos[self.cube_blue_body_id] if self.pick_blue else self.sim.data.body_xpos[self.cube_wood_body_id]
        
        region_pos = self.sim.data.body_xpos[self.serving_region_red_body_id] if self.place_on_red else \
                            self.sim.data.body_xpos[self.serving_region_green_body_id]
        
        dist_to_region = np.abs(cube_pos - region_pos)

        cube = self.cube_blue if self.pick_blue else self.cube_wood

        contact = self.check_contact(cube, self.serving_region_red) if self.place_on_red else \
                    self.check_contact(cube, self.serving_region_green)
        return np.all(dist_to_region < 0.05) and contact
    
    def _check_failure(self):
        correct_cube_pos = self.sim.data.body_xpos[self.cube_blue_body_id] if self.pick_blue else self.sim.data.body_xpos[self.cube_wood_body_id]
        wrong_cube_pos = self.sim.data.body_xpos[self.cube_wood_body_id] if self.pick_blue else self.sim.data.body_xpos[self.cube_blue_body_id]

        correct_region_pos = self.sim.data.body_xpos[self.serving_region_red_body_id] if self.place_on_red else self.sim.data.body_xpos[self.serving_region_green_body_id]
        wrong_region_pos = self.sim.data.body_xpos[self.serving_region_green_body_id] if self.place_on_red else self.sim.data.body_xpos[self.serving_region_red_body_id]

        wrong_cube_to_correct_region_dist = np.linalg.norm(wrong_cube_pos - correct_region_pos)
        correct_cube_to_wrong_region_dist = np.linalg.norm(correct_cube_pos - wrong_region_pos)

        correct_cube = self.cube_blue if self.pick_blue else self.cube_wood
        wrong_cube = self.cube_wood if self.pick_blue else self.cube_blue

        correct_region = self.serving_region_red if self.place_on_red else self.serving_region_green
        wrong_region = self.serving_region_green if self.place_on_red else self.serving_region_red

        return (self.check_contact(correct_cube, wrong_region) and correct_cube_to_wrong_region_dist < 0.05) or \
                (self.check_contact(wrong_cube, correct_region) and wrong_cube_to_correct_region_dist < 0.05)

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cube_blue)

    def _check_grasp_tolerant(self, gripper, object_geoms):
        """
        Tolerant version of check grasp function - often needed for checking grasp with Shapenet mugs.

        TODO: only tested for panda, update for other robots.
        """
        check_1 = self._check_grasp(gripper=gripper, object_geoms=object_geoms)
        check_2 = self._check_grasp(gripper=["gripper0_finger1_collision", "gripper0_finger2_pad_collision"], object_geoms=object_geoms)
        check_3 = self._check_grasp(gripper=["gripper0_finger2_collision", "gripper0_finger1_pad_collision"], object_geoms=object_geoms)

        return check_1 or check_2 or check_3
    
    def get_processed_obs(self):
        obs = self._get_observations(force_update=True)
        for k in obs.keys():
            if 'image' in k:
                obs[k] = np.flip(cv2.resize(cv2.cvtColor(obs[k], cv2.COLOR_RGB2BGR).astype(float), (84, 84)), 0)
        obs = self.append_extra_obs(obs)

        return obs
    
    def append_extra_obs(self, obs):
        obs['task_obs'] = np.r_[self.cam_angles, obs['stage_id']]
        obs['camera_angle'] = np.array(self.cam_angles)  # only for task usage
        obs['camera_pos'] = np.array(self.cam_pos)
        obs['is_policy_uncertain'] = np.zeros((1,))
        return obs
    
    def record_obs(self):
        obs = self._get_observations(force_update=True)
        for k in obs.keys():
            if 'image' in k:
                obs[k] = np.flip(cv2.cvtColor(obs[k], cv2.COLOR_RGB2BGR).astype(float), 0)
        self.recorded_obs.append(obs)
    
    # this function steps the camera
    def rotate_camera(self, camera_action):
        self.total_env_steps_taken += 1 
        # cam angles in rad
        action = np.zeros(2)
        pos_action = np.zeros(3)
        if camera_action == 1:
            action[0] = 1
        elif camera_action == 2:
            action[0] = -1
        elif camera_action == 3:
            action[1] = 1
        elif camera_action == 4:
            action[1] = -1
        elif camera_action == 5:
            pos_action[0] = 0.1
        elif camera_action == 6:
            pos_action[0] = -0.1
        elif camera_action == 7:
            pos_action[1] = 0.1
        elif camera_action == 8:
            pos_action[1] = -0.1
        # elif camera_action == 9:
        #     pos_action[2] = 0.1
        # elif camera_action == 10:
        #     pos_action[2] = -0.1

        self.cam_angles += action * 10. * np.pi / 180.
        self.cam_angles = np.clip(self.cam_angles, -1, 1)

        # first action dim is pan 
        pan_angle = self.cam_angles[0]
        pan_euler = [0,0, pan_angle]
        pan_rot = T.euler2mat(pan_euler)
        pan_mat = T.make_pose(np.zeros(3), pan_rot)

        # second action dim is tilt 
        tilt_angle = self.cam_angles[1]
        tilt_euler = [0,tilt_angle,0]
        tilt_rot = T.euler2mat(tilt_euler)
        tilt_mat = T.make_pose(np.zeros(3), tilt_rot)

        # third action dim is movement
        self.cam_pos += pos_action
        self.cam_pos = np.clip(self.cam_pos, [0, -1, 0], [0.4, 1, 0.4])
        position_mat = T.make_pose(self.cam_pos, np.eye(3))
        
        # new cam pose 
        new_cam_pose_mat = self.ptu_origin_mat @ pan_mat @ tilt_mat @ position_mat @ self.ptu_to_cam
        new_cam_pos, new_cam_quat = T.mat2pose(new_cam_pose_mat)
        self.cam_mover.set_camera_pose(pos=new_cam_pos, quat=new_cam_quat)

        self.record_obs()
        obs = self.get_processed_obs()
        return obs, 0, False, {}

    # this function steps the robot
    def step(self, action):
        self.total_env_steps_taken += 1
        obs, reward, done, info = super().step(action)

        obs = self.get_processed_obs()
        self.record_obs()
        return obs, reward, self._check_success() or done, info
 
    
    def reset(self):
        self.stage_id = 0
        self.total_env_steps_taken = 0
        obs = super().reset()
        
        if not self.setting_camera:
            self.setting_camera = True  
            self.cam_mover = CameraMover(
                env=self,
                camera=self.CAMERA_NAME,
            )
            self.setting_camera = False
            
            self.cam_angles = np.array([0.,0.]) # pan, tilt
            self.cam_pos = np.array([0.,0.,0.])
            cam_origin_pos, ori = self.cam_mover.get_camera_pose()
            cam_origin_mat = T.make_pose(cam_origin_pos, T.quat2mat(ori))
            self.ptu_origin_mat = T.make_pose(cam_origin_pos, np.eye(3))
            self.ptu_to_cam = np.linalg.inv(self.ptu_origin_mat) @ cam_origin_mat
            obs = self.get_processed_obs()

            self.recorded_obs = []
            self.record_obs()
        
        # This only needs to be called once since the information do not change
        if self.pi_format == "clip":
            pi_text = self.generate_pi_text()
            self.current_text_embeds = self.get_text_embeddings(pi_text)
       
        return obs

class OracleWalledMultiStage(WalledMultiStage, OracleBase):

    def __init__(self, **kwargs):
        WalledMultiStage.__init__(self, **kwargs)
        OracleBase.__init__(self, self.action_spec)

    def reset_skills(self):
        super().reset_skills()

        # set the first skill to start
        self.cur_skill = 'pick_cube_blue' if self.pick_blue else 'pick_cube_wood'

    def reset(self):
        obs = super().reset()
        self.reset_skills()
    
        return obs
    
    def create_skills(self):
        self.skills = {
            'pick_cube_blue': [
                ['move', 'cube_blue_pos', np.array([0, 0, 0.05])],
                ['move', 'cube_blue_pos', np.array([0, 0, 0])],
                ['grasp', [0, 0, 0, 1], 5],
            ],

            'pick_cube_wood': [
                ['move', 'cube_wood_pos', np.array([0, 0, 0.05])],
                ['move', 'cube_wood_pos', np.array([0, 0, 0])],
                ['grasp', [0, 0, 0, 1], 5],
            ],

            'place_on_red': [
                ['move', 'delta', np.array([0, 0, 0.05])],
                ['move', 'serving_region_red_pos', np.array([0, 0, 0.05])],
                ['grasp', [0, 0, 0, -1], 5],
            ],

            'place_on_green': [
                ['move', 'delta', np.array([0, 0, 0.05])],
                ['move', 'serving_region_green_pos', np.array([0, 0, 0.05])],
                ['grasp', [0, 0, 0, -1], 5],
            ],
        }

    def update_task_progress(self):
        cube = self.cube_blue if self.pick_blue else self.cube_wood
        if self._check_grasp(self.robots[0].gripper, cube.contact_geoms):
            if self.place_on_red:
                self.set_task('place_on_red')
            else:
                self.set_task('place_on_green')


if __name__=="__main__":
    from l2l.config.env.robosuite.walled_multi_stage import env_config
    import robosuite as suite
    import cv2

    from l2l.wrappers.robosuite_wrappers import RobosuiteGymWrapper, EncoderRewardWrapper

    # initialize the task
    # env = EncoderRewardWrapper(RobosuiteGymWrapper(suite.make(**env_config)))
    env = RobosuiteGymWrapper(suite.make(**env_config))
    obs, _ = env.reset()

    # do visualization
    steps = 0
    for i in range(10000):
        robot_action = np.zeros(4)#env.get_optimal_action(obs)
        # robot_action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(robot_action)

        # camera_action = int(input('enter_act'))
        for i in range(2):
            camera_action = np.random.randint(0, 9)
            obs, _, _, _ = env.unwrapped.rotate_camera(camera_action)

        steps += 1

        # print(obs['camera_angle'])
        img = np.concatenate((obs['activeview_image'], obs['agentview_image']), axis=1).astype(np.float32)/255
        cv2.imshow('img', img)
        cv2.waitKey(5)
        # input()
        env.render()
        # if i % 20 == 0:
        #     camera_action = int(input())

        if done:
            env.reset()
            steps = 0