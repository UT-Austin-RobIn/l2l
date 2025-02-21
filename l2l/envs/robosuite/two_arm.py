from collections import OrderedDict

import cv2
import numpy as np
import xml.etree.ElementTree as ET

from gymnasium import spaces

from robosuite.environments.manipulation.two_arm_env import TwoArmEnv
from robosuite.models.arenas import MultiTableArena
from robosuite.models.objects import BoxObject, CylinderObject, RoundNutObject, SquareNutObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.utils.transform_utils import convert_quat
import robosuite.utils.transform_utils as T
from robosuite.utils.camera_utils import CameraMover

from l2l.envs.robosuite.oracle_base import OracleBase
from copy import deepcopy

from robosuite.utils.mjcf_utils import CustomMaterial, add_material, find_elements, string_to_array
import mimicgen_envs
from mimicgen_envs.models.robosuite.objects import DrawerObject, LongDrawerObject, CoffeeMachinePodObject

class TwoArm(TwoArmEnv):

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
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
        self.table_offset = np.array([(0, -0.3, 0.8), (0, -10, 0.8), (0, 0.6, 0.8)])

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        self.stage_id = 0

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
        self.total_env_steps_taken = 0

        self.recorded_obs = []

    @property
    def action_space(self):
        low, high = self.action_spec

        # HACK
        low = low[4:]
        high = high[4:]
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
        xpos[1] = self.table_offset[0][1]
        self.robots[0].robot_model.set_base_xpos(xpos)
        xpos = list(self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0]))        
        xpos[1] = self.table_offset[2][1]
        arm2_pos = np.array(xpos)
        arm2_ori = [0,0,0]
        self.robots[1].robot_model.set_base_xpos(arm2_pos)
        self.robots[1].robot_model.set_base_ori(arm2_ori)

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
            quat=[0.653, 0.271, 0.271, 0.653],#[0.581, 0.405, 0.405, 0.581]
        )

        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.5, -0.3, 1.35],
            quat=[0.653, 0.271, 0.271, 0.653]
        )

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
        
        self.place_on_blue = np.random.rand() >= 0.5

        # re-name textures for Drawer
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="MatRedWood",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        ceramic = CustomMaterial(
            texture="Ceramic",
            tex_name="ceramic",
            mat_name="MatCeramic",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        lightwood = CustomMaterial(
            texture="WoodLight",
            tex_name="lightwood",
            mat_name="MatLightWood",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "3 3", "specular": "0.4","shininess": "0.1"}
        )

        self.nut = RoundNutObject("RoundNutObject")
        self.peg1 = CylinderObject(
            "Peg1Object", 
            size=(0.02, 0.1), 
            joints=None, 
            material=bluewood
        )
        self.peg2 = CylinderObject(
            "Peg2Object", 
            size=(0.02, 0.1), 
            joints=None, 
            material=greenwood
        )

        self.drawer1 = LongDrawerObject(name="Drawer1Object")
        obj_body = self.drawer1
        for material in [redwood, ceramic, lightwood]:
            tex_element, mat_element, _, used = add_material(root=obj_body.worldbody,
                                                             naming_prefix=obj_body.naming_prefix,
                                                             custom_material=deepcopy(material))
            obj_body.asset.append(tex_element)
            obj_body.asset.append(mat_element)


        self.drawer2 = LongDrawerObject(name="Drawer2Object")
        obj_body = self.drawer2
        for material in [redwood, ceramic, lightwood]:
            tex_element, mat_element, _, used = add_material(root=obj_body.worldbody,
                                                             naming_prefix=obj_body.naming_prefix,
                                                             custom_material=deepcopy(material))
            obj_body.asset.append(tex_element)
            obj_body.asset.append(mat_element)

        self.cube_reference_place = BoxObject(
            name="cube_reference_place",
            size_min=[0.025, 0.025, 0.025],
            size_max=[0.025, 0.025, 0.025],
            rgba=(0, 0, 0, 1),
            material=bluewood if self.place_on_blue else greenwood,
        )
        cube_reference_place_bounds = dict(
                x=(-0.0, -0.0),
                y=(-0.0, -0.0),
                z_rot=(-np.pi / 6., -np.pi / 6.),
                # put vertical
                # z_rot=(-np.pi / 2., -np.pi / 2.),
                reference=[0,0,0],
            )

        self.cube_blue = BoxObject(
            name="cube_blue",
            size_min=[0.02, 0.02, 0.02],
            size_max=[0.02, 0.02, 0.02],
            rgba=[0, 0, 0, 1],
            material=redwood,
        )

        self.cube_wood = BoxObject(
            name="cube_wood",
            size_min=[0.02, 0.02, 0.02],
            size_max=[0.02, 0.02, 0.02],
            rgba=[0, 0, 0, 1],
            material=darkwood,
        )
        self.placement_initializer = self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="ObjectSampler-drawer1",
                mujoco_objects=self.drawer1,
                x_range=[-0., 0.],
                y_range=[-0., 0.],
                rotation=[0.,0.],
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=False,
                reference_pos=self.table_offset[2] + [-0.1, 0.2, 0.],
                z_offset=0.03,
        ))
        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="ObjectSampler-drawer2",
                mujoco_objects=self.drawer2,
                x_range=[-0., 0.],
                y_range=[-0., 0.],
                rotation=[0.,0.],
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=False,
                reference_pos=self.table_offset[2] + [0.15, 0.2, 0.],
                z_offset=0.03,
        ))

        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="ObjectSampler-cube_blue",
                mujoco_objects=self.cube_blue,
                x_range=[-0.1, 0.1],
                y_range=[-0.1, 0.1],
                rotation=[0,0],
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset[2] + [0, 0.2, 0.],
                z_offset=0.175,
        ))
        
        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="ObjectSampler-cube_wood",
                mujoco_objects=self.cube_wood,
                x_range=[-0.1, 0.1],
                y_range=[-0.1, 0.1],
                rotation=[0,0],
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset[2] + [0, 0.2, 0.],
                z_offset=0.175,
        ))

        self.pod_placement_initializer = UniformRandomSampler(
            name="CoffeePodInDrawerSampler",
            mujoco_objects=self.cube_reference_place,
            x_range=cube_reference_place_bounds["x"],
            y_range=cube_reference_place_bounds["y"],
            rotation=cube_reference_place_bounds["z_rot"],
            rotation_axis='z',
            # ensure_object_boundary_in_range=True, # make sure pod fits within the box
            ensure_object_boundary_in_range=False, # make sure pod fits within the box
            ensure_valid_placement=True,
            reference_pos=cube_reference_place_bounds["reference"],
            z_offset=0.,
        )

        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="ObjectSampler-nut",
                mujoco_objects=self.nut,
                x_range=[0.0, 0.0],
                y_range=[0.0, 0.0],#[-0.15, 0.15],
                rotation=[np.pi, np.pi],
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset[0]+[-0.1,0,0],
                z_offset=-0.01,
        ))

        # mujoco_objects = [self.cube_blue, self.cube_wood, self.serving_region_blue, self.serving_region_green, self.cube_blue, self.drawer1, self.drawer2, self.cube_reference_place]
        mujoco_objects = [self.nut, self.peg1, self.peg2, self.drawer1, self.drawer2, self.cube_reference_place, self.cube_blue, self.cube_wood]
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
        self.nut_id = self.sim.model.body_name2id(self.nut.root_body)
        self.peg1_body_id = self.sim.model.body_name2id(self.peg1.root_body)
        self.peg2_body_id = self.sim.model.body_name2id(self.peg2.root_body)
        self.drawer1_body_id = self.sim.model.body_name2id(self.drawer1.root_body)
        self.drawer2_body_id = self.sim.model.body_name2id(self.drawer2.root_body)
        self.cube_blue_body_id = self.sim.model.body_name2id(self.cube_blue.root_body)
        self.cube_wood_body_id = self.sim.model.body_name2id(self.cube_wood.root_body)
        self.cube_reference_place_id = self.sim.model.body_name2id(self.cube_reference_place.root_body)

        self.drawer1_qpos_addr = self.sim.model.get_joint_qpos_addr(self.drawer1.joints[0])
        self.drawer1_bottom_geom_id = self.sim.model.geom_name2id("Drawer1Object_drawer_bottom")

        self.drawer2_qpos_addr = self.sim.model.get_joint_qpos_addr(self.drawer2.joints[0])
        self.drawer2_bottom_geom_id = self.sim.model.geom_name2id("Drawer2Object_drawer_bottom")


    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        self.sim.data.qpos[self.drawer1_qpos_addr] = 0.
        self.sim.data.qpos[self.drawer2_qpos_addr] = 0.

        self.sim.model.body_pos[self.peg1_body_id] = self.table_offset[0] + [0.1, -0.1, 0.05]
        self.sim.model.body_pos[self.peg2_body_id] = self.table_offset[0] + [0.1, 0.1, 0.05]

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                if obj is self.drawer1 or obj is self.drawer2:
                    # object is fixture - set pose in model
                    body_id = self.sim.model.body_name2id(obj.root_body)
                    obj_pos_to_set = np.array(obj_pos)
                    obj_pos_to_set[2] = 0.805 # hardcode z-value to make sure it lies on table surface
                    self.sim.model.body_pos[body_id] = obj_pos_to_set
                    self.sim.model.body_quat[body_id] = obj_quat
                else:
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

            self.sim.forward()

            # sample pod location relative to center of drawer bottom geom surface
            # https://github.com/NVlabs/mimicgen_environments/blob/45db4b35a5a79e82ca8a70ce1321f855498ca82c/mimicgen_envs/envs/robosuite/coffee.py#L21
            cube_reference_place_placement = self.pod_placement_initializer.sample(on_top=False)
            assert len(cube_reference_place_placement) == 1
            rel_pod_pos, rel_pod_quat, pod_obj = list(cube_reference_place_placement.values())[0]
            rel_pod_pos, rel_pod_quat = np.array(rel_pod_pos), np.array(rel_pod_quat)
            assert pod_obj is self.cube_reference_place

            # center of drawer bottom
            # sample which drawer
            self.in_drawer1 = np.random.rand() >= 0.5 
            drawer_root_body = self.drawer1.root_body if self.in_drawer1 else self.drawer2.root_body
            bottom_id = self.drawer1_bottom_geom_id if self.in_drawer1 else self.drawer2_bottom_geom_id
            drawer_bottom_geom_pos = np.array(self.sim.data.geom_xpos[bottom_id])

            # our x-y relative position is sampled with respect to drawer geom frame. Here, we use the drawer's rotation
            # matrix to convert this relative position to a world relative position, so we can add it to the drawer world position
            drawer_rot_mat = T.quat2mat(T.convert_quat(self.sim.model.body_quat[self.sim.model.body_name2id(drawer_root_body)], to="xyzw"))
            rel_pod_pos[:2] = drawer_rot_mat[:2, :2].dot(rel_pod_pos[:2])

            # also convert the sampled pod rotation to world frame
            rel_pod_mat = T.quat2mat(T.convert_quat(rel_pod_quat, to="xyzw"))
            pod_mat = drawer_rot_mat.dot(rel_pod_mat)
            pod_quat = T.convert_quat(T.mat2quat(pod_mat), to="wxyz")

            # get half-sizes of drawer geom and coffee pod to place coffee pod at correct z-location (on top of drawer bottom geom)
            drawer_bottom_geom_z_offset = self.sim.model.geom_size[bottom_id][-1] # half-size of geom in z-direction
            cube_reference_place_bottom_offset = np.abs(self.cube_reference_place.bottom_offset[-1])
            cube_reference_place_z = drawer_bottom_geom_pos[2] + drawer_bottom_geom_z_offset + cube_reference_place_bottom_offset + 0.001

            # set coffee pod in center of drawer
            pod_pos = np.array(drawer_bottom_geom_pos) + rel_pod_pos
            pod_pos[-1] = cube_reference_place_z

            self.sim.data.set_joint_qpos(pod_obj.joints[0], np.concatenate([np.array(pod_pos), np.array(pod_quat)]))
            self.sim.forward()
    
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
            def cube_wood_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube_wood_body_id])

            @sensor(modality=modality)
            def nut_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.nut_id])

            @sensor(modality=modality)
            def peg1_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.peg1_body_id])
           
            @sensor(modality=modality)
            def peg2_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.peg2_body_id])

            @sensor(modality=modality)
            def drawer1_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.drawer1_body_id])
            
            @sensor(modality=modality)
            def drawer2_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.drawer2_body_id])
            
            @sensor(modality=modality)
            def privileged_info(obs_cache):
                info = np.zeros(2).astype(np.int8)
                if self.place_on_blue:
                    info[0] = 1
                else:
                    info[1] = 1
                return info
            
            @sensor(modality="extra")
            def stage_id(obs_cache):
                return np.array([0])
                
            @sensor(modality='extra')
            def task_obs(obs_cache):
                return np.zeros(3)
            
            @sensor(modality='extra')
            def camera_angle(obs_cache):
                return np.zeros((2,))

            # @sensor(modality="extra")
            # def is_policy_uncertain(obs_cache):
            #     return np.zeros((1,))

            sensors = [nut_pos, peg1_pos, peg2_pos, drawer1_pos, drawer2_pos, cube_blue_pos, cube_wood_pos, privileged_info, stage_id, task_obs, camera_angle]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def on_peg(self, peg_body_id):

        obj_pos = np.array(self.sim.data.body_xpos[self.nut_id])
        peg_pos = np.array(self.sim.data.body_xpos[peg_body_id])
        res = False
        close_x = abs(obj_pos[0] - peg_pos[0]) < 0.03
        close_y = abs(obj_pos[1] - peg_pos[1]) < 0.03
        close_z = obj_pos[2] < self.table_offset[0][2] + 0.05
        
        if close_x and close_y and close_z:
            res = True
        return res

    def _check_success(self):
        """
        Check if blocks are stacked correctly.

        Returns:
            bool: True if blocks are correctly stacked
        """
        if self.place_on_blue:
            peg_id = self.peg1_body_id
        else:
            peg_id = self.peg2_body_id
        return self.on_peg(peg_id)
    
    def _check_failure(self):
        """
            Check if blocks are stacked incorrectly.
        """
        if self.place_on_blue:
            peg_id = self.peg2_body_id
        else:
            peg_id = self.peg1_body_id
        return self.on_peg(peg_id)

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
        if camera_action == 1:
            action[0] = 1
        elif camera_action == 2:
            action[0] = -1
        elif camera_action == 3:
            action[1] = 1
        elif camera_action == 4:
            action[1] = -1

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
        
        # new cam pose 
        new_cam_pose_mat = self.ptu_origin_mat @ pan_mat @ tilt_mat @ self.ptu_to_cam
        new_cam_pos, new_cam_quat = T.mat2pose(new_cam_pose_mat)
        self.cam_mover.set_camera_pose(pos=new_cam_pos, quat=new_cam_quat)

        obs = self.get_processed_obs()
        obs = self.append_extra_obs(obs)
        self.record_obs()
        return obs, 0, False, {}

    # this function steps the robot
    def step(self, action):
        self.total_env_steps_taken += 1
        # hack to zero action for second arm 
        robot0_action = action
        super_action = np.zeros(8)
        super_action[:4] = robot0_action
        obs, reward, done, info = super().step(super_action)

        obs = self.get_processed_obs()
        self.record_obs()
        return obs, reward, self._check_success() or done, info

    # this function steps the second robot
    def step2(self, action):
        self.total_env_steps_taken += 1

        # hack to zero action for first arm 
        robot1_action = action
        super_action = np.zeros(8)
        super_action[4:] = robot1_action
        obs, reward, done, info = super().step(super_action)

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
            cam_origin_pos, ori = self.cam_mover.get_camera_pose()
            cam_origin_mat = T.make_pose(cam_origin_pos, T.quat2mat(ori))
            self.ptu_origin_mat = T.make_pose(cam_origin_pos, np.eye(3))
            self.ptu_to_cam = np.linalg.inv(self.ptu_origin_mat) @ cam_origin_mat
            obs = self.get_processed_obs()

            self.recorded_obs = []
            self.record_obs()
       
        return obs

class TwoArmOracle(TwoArm, OracleBase):

    def __init__(self, **kwargs):
        TwoArm.__init__(self, **kwargs)
        OracleBase.__init__(self, self.action_spec)

    def reset_skills(self):
        super().reset_skills()
        self.cur_skill = 'move_to_nut'

    def reset(self):
        obs = super().reset()
        self.reset_skills()
        return obs

    def create_skills(self):
        self.skills = OrderedDict()

        self.skills['move_to_nut'] = [
            ['move', 'nut_pos', np.array([0.075, 0, 0.05])],
            ['move', 'nut_pos', np.array([0.075, 0, -0.001])],
        ]
        
        self.skills['pick_nut'] = [
                ['grasp', [0, 0, 0, 1], 5],
                ['move', 'delta', np.array([0, 0, 0.2])],
            ]
        
        self.skills['move_to_blue'] = [
                ['move', 'peg1_pos', np.array([0.075, 0, 0.2])],
                ['move', 'peg1_pos', np.array([0.075, 0, 0.0])]
            ]
        
        self.skills['move_to_green'] = [
                ['move', 'peg2_pos', np.array([0.075, 0, 0.2])],
                ['move', 'peg2_pos', np.array([0.075, 0, 0.0])]
            ]
        
        self.skills['place_block'] = [
                ['grasp', [0, 0, 0, -1], 5],
            ]
        
        self.skills["wait"] = [
            ["wait", None, 1],
        ]


    def update_task_progress(self):
        # cube = self.cube_blue if self.pick_blue else self.cube_wood
        # if self._check_grasp(self.robots[0].gripper, self.nut.contact_geoms):
        if self.place_on_blue:
            self.set_task('place_on_blue')
        else:
            self.set_task('place_on_green')

if __name__=="__main__":
    from l2l.config.env.robosuite.two_arm import env_config
    import robosuite as suite
    import cv2

    from l2l.wrappers.robosuite_wrappers import RobosuiteGymWrapper

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
        env.render()

        camera_action = int(input('enter_act'))
        for i in range(2):
            # camera_action = np.random.choice([0, 1, 2, 3, 4])
            obs, _, _, _ = env.unwrapped.rotate_camera(camera_action)



        steps += 1

        print(obs['camera_angle'])
        img = obs['activeview_image'].astype(np.float32)/255
        cv2.imshow('img', img)
        cv2.waitKey(5)
        # input()
        # env.render()
        # if i % 20 == 0:
        #     camera_action = int(input())

        if done:
            env.reset()
            steps = 0