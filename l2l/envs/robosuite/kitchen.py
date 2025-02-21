from collections import OrderedDict

import cv2
import numpy as np

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import MultiTableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.utils.transform_utils import convert_quat

from l2l.envs.robosuite.oracle_base import OracleBase
from l2l.envs.robosuite.kitchen_objects import TargetObject, ButtonObject, StoveObject, PotObject

from robosuite.utils.mjcf_utils import CustomMaterial, array_to_string, find_elements, add_material
from robosuite.models.objects import CylinderObject, BoxObject, BallObject, CompositeObject
from copy import deepcopy

import robosuite.utils.transform_utils as T
import xml.etree.ElementTree as ET
from robosuite.utils.camera_utils import CameraMover

from gymnasium import spaces
import l2l
import os
from robosuite.models.objects import BreadObject
import torch


class KitchenEnv(SingleArmEnv):

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
            pi_format='onehot',
            text_encoder=None,
            partitioned_encoding=False,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array([(0, 0.8, 0.8), (0, -0.8, 0.8), (0, 0, 0.8)])
        self.mid_table_idx = 2
        self.right_table_idx = 0
        self.left_table_idx = 1

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        self.fixtures_dict = {}
        self.objects_dict = {}

        # size of stage info vec      
        self.stage_id = 0
        self.correct_cook_wait_action = False

        #################  language encoding related  #################
        self.pi_format = pi_format
        self.text_encoder = text_encoder
        self.partitioned_encoding = partitioned_encoding

        self.pi_to_text = {}
        self.pi_to_text['pick_bread'] = [
                                        "Lift up the bread.",
                                        # "Pick up the bread",
                                        # "Lift the bread",
                                        # "Grasp the bread"
                                    ]
        self.pi_to_text['pick_meat'] = [
                                        # "Lift up the meat",
                                        # "Pick up the meat",
                                        # "Lift the meat",
                                        "Grasp the meat."
                                    ]

        self.pi_to_text["5min"] = [
            "Cook for a short amount of time."
        ]
        self.pi_to_text["1h"] = [
            "Cook until it is well-done."
        ]

        self.pi_to_text['place_on_red'] = [
                                            "Place the pot on the red region.",
                                            # "Put the pot on the red region",
                                            # "Place the pot on the red area",
                                            # "Put the pot on the red area"
                                        ]
        self.pi_to_text['place_on_green'] = [
                                            # "Place the pot on the green region",
                                            # "Put the pot on the green region",
                                            # "Place the pot on the green area",
                                            "Put the pot on the green area."
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

        self.n_camera_actions = 5
        self.total_env_steps_taken = 0

        self.recorded_obs = []

    def generate_pi_text(self):
        """
        Generate the text input for the text encoder based on the privileged info
        """

        if self.pick_bread:
            s1 = np.random.choice(self.pi_to_text['pick_bread'])
        else:
            s1 = np.random.choice(self.pi_to_text['pick_meat'])

        if self.cook_time == 0:
            s2 = np.random.choice(self.pi_to_text['5min'])
        else:
            s2 = np.random.choice(self.pi_to_text['1h'])

        if self.place_on_red:
            s3 = np.random.choice(self.pi_to_text['place_on_red'])
        else:
            s3 = np.random.choice(self.pi_to_text['place_on_green'])

        if self.partitioned_encoding:
            return [s1, s2, s3]
        else:
            return [s1 + " and " + s2 + " and " + s3]

    @torch.no_grad()
    def get_text_embeddings(self, text):
        self.pi_text = text
        inputs = self.text_encoder[0](text, padding=True, return_tensors="pt")
        outputs = self.text_encoder[1](**inputs)
        text_embeds = outputs.text_embeds
        text_embeds = torch.flatten(text_embeds)
        return text_embeds

    @property
    def action_space(self):
        low, high = self.action_spec
        return spaces.Box(low, high)

    def staged_rewards(self):
        """
        Helper function to calculate staged rewards based on current physical states.

        Returns:
            3-tuple:

                - (float): reward for reaching and grasping
                - (float): reward for lifting and aligning
                - (float): reward for stacking
        """
        return 0, 0, 0

    def _load_custom_material(self):
        """
        Define all the textures
        """
        self.custom_material_dict = dict()

        tex_attrib = {
            "type": "cube"
        }

        self.custom_material_dict["bread"] = CustomMaterial(
            texture="Bread",
            tex_name="bread",
            mat_name="MatBread",
            tex_attrib=tex_attrib,
            mat_attrib={"texrepeat": "3 3", "specular": "0.4", "shininess": "0.1"}
        )
        self.custom_material_dict["darkwood"] = CustomMaterial(
            texture="WoodDark",
            tex_name="darkwood",
            mat_name="MatDarkWood",
            tex_attrib=tex_attrib,
            mat_attrib={"texrepeat": "3 3", "specular": "0.4", "shininess": "0.1"}
        )

        self.custom_material_dict["lightwood"] = CustomMaterial(
            texture="WoodLight",
            tex_name="lightwood",
            mat_name="MatLightWood",
            tex_attrib=tex_attrib,
            mat_attrib={"texrepeat": "3 3", "specular": "0.4", "shininess": "0.1"}
        )

        self.custom_material_dict["metal"] = CustomMaterial(
            texture="Metal",
            tex_name="metal",
            mat_name="MatMetal",
            tex_attrib=tex_attrib,
            mat_attrib={"specular": "1", "shininess": "0.3", "rgba": "0.9 0.9 0.9 1"}
        )

        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1"
        }

        self.custom_material_dict["greenwood"] = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.custom_material_dict["redwood"] = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="MatRedWood",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        self.custom_material_dict["bluewood"] = CustomMaterial(
            texture="WoodBlue",
            tex_name="bluewood",
            mat_name="MatBlueWood",
            tex_attrib=tex_attrib,
            mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
        )

        self.custom_material_dict["lemon"] = CustomMaterial(
            texture="Lemon",
            tex_name="lemon",
            mat_name="MatLemon",
            tex_attrib=tex_attrib,
            mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
        )

        self.custom_material_dict["steel"] = CustomMaterial(
            texture="SteelScratched",
            tex_name="steel_scratched_tex",
            mat_name="MatSteelScratched",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

    def _load_fixtures_in_arena(self, mujoco_arena):
        self.table_body = mujoco_arena.worldbody.find(f"./body[@name='table{self.mid_table_idx}']")

        self.fixtures_dict["button"] = ButtonObject(name="button")
        button_object = self.fixtures_dict["button"].get_obj()
        button_object.set("pos", array_to_string((0, 0, 0.02)))
        button_object.set("quat", array_to_string((0., 0., 0., 1.)))
        self.table_body.append(button_object)

        self.fixtures_dict["stove"] = StoveObject(name="stove")
        stove_object = self.fixtures_dict["stove"].get_obj()
        stove_object.set("pos", array_to_string((0, 0, 0.003)))
        self.table_body.append(stove_object)

        # self.fixtures_dict["target"] = TargetObject(name="target")
        # target_object = self.fixtures_dict["target"].get_obj()
        # target_object.set("pos", array_to_string((0, 0, 0.003)))
        # self.table_body.append(target_object)

    def _load_objects_in_arena(self, mujoco_arena):
        self.objects_dict["pot"] = PotObject(name="pot")


        butter_size = [0.02, 0.02, 0.02]
        self.objects_dict["butter"] = BreadObject(name="butter")
        # BoxObject(
        #     name="butter",
        #     size_min=butter_size,
        #     size_max=butter_size,
        #     rgba=[1, 0, 0, 1],
        #     material=self.custom_material_dict["lemon"],
        #     density=100.,
        # )
        meatball_size = [0.02, 0.02, 0.02]
        # Change everything to box
        self.objects_dict["meatball"] = BoxObject(
            name="meatball",
            size_min=meatball_size,
            size_max=meatball_size,
            rgba=[1, 0, 0, 1],
            material=self.custom_material_dict["darkwood"],
            density=100.,
        )

    def _setup_placement_initializer(self):

        self.placement_initializer = self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="ObjectSampler-cubeA",
                mujoco_objects=self.food_menu,
                x_range=[0.01, 0.01],
                y_range=[0.01, 0.01],  # [-0.15, 0.15],
                rotation=(0., 0.),
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset[self.right_table_idx],
                z_offset=0.01,
            ))

        serving_x = 0.2
        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="ObjectSampler-serving_region_red",
                mujoco_objects=self.serving_region_red,
                x_range=[serving_x, serving_x],
                y_range= [0.05, 0.05],  # [-0.15, 0.15],
                rotation=(0., 0.),
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset[self.mid_table_idx],
                z_offset=0.01,
            ))

        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="ObjectSampler-serving_region_green",
                mujoco_objects=self.serving_region_green,
                x_range=[serving_x, serving_x],
                y_range=[0.25, 0.25],  # [-0.15, 0.15],
                rotation=(0., 0.),
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset[self.mid_table_idx],
                z_offset=0.01,
            ))

        if self.place_on_red:
            checkbox_y = 0.05
        else:
            checkbox_y = 0.25
        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="ObjectSampler-checkbox_region",
                mujoco_objects=self.checkbox_region,
                x_range=[serving_x + 0.13, serving_x + 0.13],
                y_range=[checkbox_y, checkbox_y],  # [-0.15, 0.15],
                rotation=(0., 0.),
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset[self.mid_table_idx],
                z_offset=0.02,
            ))


        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="ObjectSampler-clock",
                mujoco_objects=self.clock_time,
                x_range=[0.0, 0.0],
                y_range=[-0.0, 0.0],
                rotation=None, #(0.01, 0.02),
                # rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset[self.left_table_idx],
                z_offset=0.01,
            ))

        self.butter_x_range = [-0.05, -0.05]
        self.butter_y_range = [-0.3, -0.3] # originally -0.2 to -0.3
        self.meatball_x_range = [-0.05, -0.05]
        self.meatball_y_range = [-0.2, -0.2]
        self.pot_x_range = [-0.0, -0.0]
        self.pot_y_range = [-0.00, -0.00] # originally -0.1 to 0

        # initializer for added objects
        self.placement_initializer.append_sampler(
        sampler=UniformRandomSampler(
            name="ObjectSampler-butter",
            mujoco_objects=self.objects_dict["butter"],
            x_range=self.butter_x_range,
            y_range=self.butter_y_range,
            rotation=(-np.pi / 2., -np.pi / 2.),
            rotation_axis='z',
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset[self.mid_table_idx],
            z_offset=0.01,
        ))

        self.placement_initializer.append_sampler(
        sampler=UniformRandomSampler(
            name="ObjectSampler-meatball",
            mujoco_objects=self.objects_dict["meatball"],
            x_range=self.meatball_x_range,
            y_range=self.meatball_y_range,
            rotation=(-np.pi / 2., -np.pi / 2.),
            rotation_axis='z',
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset[self.mid_table_idx],
            z_offset=0.01,
        ))

        self.placement_initializer.append_sampler(
        sampler=UniformRandomSampler(
            name="ObjectSampler-pot",
            mujoco_objects=self.objects_dict["pot"],
            x_range=self.pot_x_range,
            y_range=self.pot_y_range,
            rotation=(-0.1, 0.1),
            rotation_axis='z',
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset[self.mid_table_idx],
            z_offset=0.02,
        ))

    def _setup_original_objects(self, mujoco_arena):

        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.5, 0.0, 1.35],
            quat=[0.653, 0.271, 0.271, 0.653]
        )

        mujoco_arena.set_camera(
            camera_name="sideview",
            pos=[0, 0, 1.35],
            quat=[0.0043953401036560535, -0.01594211906194687, -0.3870881199836731, -0.9218944311141968]
        )

        mujoco_arena.set_camera(
            camera_name="activeview",
            pos=[0.6, 0, 1.45],
            quat=[0.653, 0.271, 0.271, 0.653],  # [0.581, 0.405, 0.405, 0.581]
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

        #########  Creating customized materials   #############

        def create_new_material(name):
            mat = CustomMaterial(
                texture="WoodGreen",
                tex_name=name,
                mat_name=f"{name}_mat",
                tex_attrib=tex_attrib,
                mat_attrib=mat_attrib,
            )
            mat.tex_attrib["file"] = os.path.join(l2l.__path__[0], f"textures/{name}.png")
            return mat

        checkmark = create_new_material("checkmark")
        clock_5min = create_new_material("timer_long")
        clock_1h = create_new_material("timer_short")
        beef_menu = create_new_material("beans_menu")
        breakfast_menu = create_new_material("breakfast")

        #########  End of customized materials   #############

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
            mat_attrib={"texrepeat": "3 3", "specular": "0.4", "shininess": "0.1"}
        )

        serve_box_size = 0.06

        self.serving_region_red = BoxObject(
            name="serving_region_red",
            size_min=[serve_box_size, serve_box_size, 0.001],
            size_max=[serve_box_size, serve_box_size, 0.001],
            material=redwood,
        )

        self.serving_region_green = BoxObject(
            name="serving_region_green",
            size_min=[serve_box_size, serve_box_size, 0.001],
            size_max=[serve_box_size, serve_box_size, 0.001],
            material=greenwood,
        )

        # self.serving_region_red = CylinderObject(
        #     name="serving_region_red",
        #     size_min=[serve_box_size, 0.0015],
        #     size_max=[serve_box_size, 0.0015],
        #     material=redwood,
        # )

        # self.serving_region_green = CylinderObject(
        #     name="serving_region_green",
        #     size_min=[serve_box_size, 0.0015],
        #     size_max=[serve_box_size, 0.0015],
        #     material=greenwood,
        # )

        self.checkbox_region = BoxObject(
            name="checkbox_region",
            size_min=[0.03, 0.03, 0.002],
            size_max=[0.03, 0.03, 0.002],
            material=checkmark,
        )

        self.place_on_red = np.random.rand() >= 0.5
        self.pick_bread = np.random.rand() >= 0.5

        time_values = 2
        self.cook_time = np.random.randint(time_values)

        clock_size = [0.12, 0.003, 0.12] # [0.08, 0.08, 0.003]
        self.clock_time = BoxObject(
            name="clock",
            size_min=clock_size,
            size_max=clock_size,
            material=clock_5min if self.cook_time == 0 else clock_1h,
            joints=None,
        )


        self.food_menu = BoxObject(
            name="cubeA",
            size_min=[0.2, 0.2, 0.003],
            size_max=[0.2, 0.2, 0.003],
            # rgba=[0, 1.0 * self.cook_time / (time_values-1), 1.0 * self.cook_time / (time_values-1), 1], # [0, 0, 0, 1]
            material=breakfast_menu if self.pick_bread else beef_menu,
        )


    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = list(self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0]))
        # xpos[1] += 0.6 # this would set the robot to the right most table
        xpos[0] += 0.1
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = MultiTableArena(
            table_full_sizes=self.table_full_size,
            table_frictions=self.table_friction,
            table_offsets=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])


        self._load_custom_material()
        self._load_fixtures_in_arena(mujoco_arena)
        self._load_objects_in_arena(mujoco_arena)
        self._setup_original_objects(mujoco_arena)

        self._setup_placement_initializer()

        mujoco_objects = [self.food_menu, self.serving_region_red, self.serving_region_green,
                          self.checkbox_region, self.clock_time]

        self.objects = list(self.objects_dict.values())
        self.fixtures = list(self.fixtures_dict.values())

        for fixture in self.fixtures:
            if issubclass(type(fixture), CompositeObject):
                continue
            for material_name, material in self.custom_material_dict.items():
                tex_element, mat_element, _, used = add_material(root=fixture.worldbody,
                                                                 naming_prefix=fixture.naming_prefix,
                                                                 custom_material=deepcopy(material))
                fixture.asset.append(tex_element)
                fixture.asset.append(mat_element)


        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=mujoco_objects + self.objects # mujoco_objects,
        )

        for fixture in self.fixtures:
            self.model.merge_assets(fixture)

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.cubeA_body_id = self.sim.model.body_name2id(self.food_menu.root_body)
        self.serving_region_red_body_id = self.sim.model.body_name2id(self.serving_region_red.root_body)
        self.serving_region_green_body_id = self.sim.model.body_name2id(self.serving_region_green.root_body)
        self.cube_reference_body_id = self.sim.model.body_name2id(self.clock_time.root_body)


        # Kitchen object ref
        self.obj_body_id = dict()

        for object_name, object_body in self.objects_dict.items():
            self.obj_body_id[object_name] = self.sim.model.body_name2id(object_body.root_body)
        for fixture_name, fixture_body in self.fixtures_dict.items():
            self.obj_body_id[fixture_name] = self.sim.model.body_name2id(fixture_body.root_body)

        self.button_qpos_addrs = self.sim.model.get_joint_qpos_addr(self.fixtures_dict["button"].joints[0])
        self.pot_right_handle_id = self.sim.model.geom_name2id('pot_handle_right_0')
        self.button_switch_pad_id = self.sim.model.geom_name2id('button_switch_pad')

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
                if obj.name == "Door":
                    door_body_id = self.sim.model.body_name2id(obj.root_body)
                    self.sim.model.body_pos[door_body_id] = obj_pos
                    self.sim.model.body_quat[door_body_id] = obj_quat
                elif obj.name == "clock":
                    clock_body_id = self.sim.model.body_name2id(self.clock_time.root_body)
                    self.sim.model.body_pos[clock_body_id] = obj_pos
                    self.sim.model.body_quat[clock_body_id] = [0.48066848, 0., 0., 0.87690239]
                else:
                    # print(obj.name)
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        self.button_x_range = [-0.05, -0.05]
        self.button_y_range = [0.2, 0.2]
        self.stove_x_range = [0.15, 0.15]
        self.stove_y_range = [-0.15, -0.15]
        self.target_x_range = [0.05, 0.25]
        self.target_y_range = [0.1, 0.2]

        # fixtures reset
        for obj_name, obj_x_range, obj_y_range in [["button", self.button_x_range, self.button_y_range],
                                                   ["stove", self.stove_x_range, self.stove_y_range],
                                                   #["target", self.target_x_range, self.target_y_range]
                                                   ]:
            obj = self.fixtures_dict[obj_name]
            body_id = self.sim.model.body_name2id(obj.root_body)
            obj_x = np.random.uniform(obj_x_range[0], obj_x_range[1])
            obj_y = np.random.uniform(obj_y_range[0], obj_y_range[1])
            obj_z = 0.005 if obj_name == "target" else 0.02
            self.sim.model.body_pos[body_id] = (obj_x, obj_y, obj_z)
            if obj_name == "button":
                self.sim.model.body_quat[body_id] = (0., 0., 0., 1.)


        # button and meatball state reset
        self.sim.data.set_joint_qpos(self.fixtures_dict["button"].joints[0], np.array([-0.3]))
        # fix this, otherwise the openloop get_optimal_skill_sequence will not work
        self.butter_melt_status = -0.5 # np.random.uniform(-1, -0.5)
        self.meatball_cook_status = -0.5 # np.random.uniform(-1, -0.5)

        self.meatball_overcooked = False

        self.button_on = False
        self.butter_in_pot = False
        self.meatball_in_pot = False
        self.pot_on_stove = False

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)

        # Check if stove is turned on or not
        self._post_process()

        return reward, done, info

    def _post_process(self):
        self.butter_in_pot = self.check_contact(self.objects_dict["butter"], "pot_body_bottom")
        self.meatball_in_pot = self.check_contact(self.objects_dict["meatball"], "pot_body_bottom")
        self.pot_on_stove = self.check_contact("stove_collision_burner", "pot_body_bottom")

        if self.sim.data.qpos[self.button_qpos_addrs] < 0.0:
            self.button_on = False
        else:
            self.button_on = True

        self.fixtures_dict["stove"].set_sites_visibility(sim=self.sim, visible=self.button_on)


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
            def cubeA_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cubeA_body_id])

            @sensor(modality=modality)
            def cubeA_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cubeA_body_id]), to="xyzw")

            @sensor(modality=modality)
            def serving_region_red_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.serving_region_red_body_id])

            @sensor(modality=modality)
            def serving_region_green_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.serving_region_green_body_id])

            @sensor(modality=modality)
            def gripper_to_cubeA(obs_cache):
                return (
                    obs_cache["cubeA_pos"] - obs_cache[f"{pf}eef_pos"]
                    if "cubeA_pos" in obs_cache and f"{pf}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            def privileged_info(obs_cache):
                if self.pi_format == 'discrete':
                    raise NotImplementedError

                elif self.pi_format == "onehot":
                    idx = 0
                    if self.place_on_red == 1:
                        idx += 1
                    if self.cook_time == 1:
                        idx += 2
                    if self.pick_bread == 1:
                        idx += 4
                    priv_info = np.zeros(8)
                    priv_info[idx] = 1

                elif self.pi_format == "clip":
                    priv_info = self.current_text_embeds

                else:
                    raise NotImplementedError


                return priv_info

            @sensor(modality=modality)
            def gt_privileged_info(obs_cache):
                return np.array([self.place_on_red, self.cook_time, self.pick_bread], dtype=float)


            @sensor(modality=modality)
            def butter_melt_status(obs_cache):
                return self.butter_melt_status # np.array([self.butter_melt_status])

            @sensor(modality=modality)
            def meatball_cook_status(obs_cache):
                return self.meatball_cook_status

            @sensor(modality=modality)
            def button_joint_qpos(obs_cache):
                return self.sim.data.qpos[self.button_qpos_addrs]

            # TODO: what is this used for?
            @sensor(modality="extra")
            def stage_id(obs_cache):
                stage_id = np.zeros(3)

                # if self.stage_id == 1:
                #     if ((self.meatball_cook_status > 0) or (self.butter_melt_status > 0)) and (not self.button_on):
                #         self.stage_id = 2
                # elif self.stage_id == 0:
                #     if self.button_on:
                #         self.stage_id = 1
                # stage_id[self.stage_id] = 1

                item_in_pot = self.butter_in_pot if self.pick_bread else self.meatball_in_pot
                obj_cooked = self.butter_melt_status > 0 if self.pick_bread else self.meatball_cook_status > 0

                if self.stage_id == 0:
                    if item_in_pot:
                        self.stage_id = 1
                elif self.stage_id == 1:
                    if item_in_pot and obj_cooked and self.correct_cook_wait_action == 1:
                        self.stage_id = 2

                stage_id[self.stage_id] = 1
                return stage_id 

            @sensor(modality='extra')
            def camera_angle(obs_cache):
                return np.zeros((2,))
            
            @sensor(modality="extra")
            def is_policy_uncertain(obs_cache):
                return np.zeros((1,))

            sensors = [butter_melt_status, meatball_cook_status,
                     privileged_info, camera_angle, stage_id, is_policy_uncertain, gt_privileged_info]
            sensors += [cubeA_pos, cubeA_quat,
                       serving_region_red_pos, serving_region_green_pos,
                       gripper_to_cubeA, button_joint_qpos]
            
            
            names = [s.__name__ for s in sensors]

            for obj in self.objects + self.fixtures:
                obj_sensors, obj_sensor_names = self._create_obj_sensors(obj_name=obj.name, modality="object")

                sensors += obj_sensors
                names += obj_sensor_names

            for obj_name in ["pot_handle", "button_handle"]:
                obj_sensors, obj_sensor_names = self._create_geom_sensors(obj_name=obj_name, modality="object")

                sensors += obj_sensors
                names += obj_sensor_names

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables


    def _create_obj_sensors(self, obj_name, modality="object"):
        """
        Helper function to create sensors for a given object. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.

        Args:
            obj_name (str): Name of object to create sensors for
            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given obj
                names (list): array of corresponding observable names
        """
        if obj_name in self.objects_dict:
            obj = self.objects_dict[obj_name]
        elif obj_name in self.fixtures_dict:
            obj = self.fixtures_dict[obj_name]
        else:
            raise NotImplementedError

        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality=modality)
        def obj_pos(obs_cache):
            pos = self.sim.data.body_xpos[self.obj_body_id[obj_name]]
            return pos

        @sensor(modality=modality)
        def obj_quat(obs_cache):
            return T.convert_quat(self.sim.data.body_xquat[self.obj_body_id[obj_name]], to="xyzw")

        @sensor(modality=modality)
        def obj_grasped(obs_cache):
            grasped = int(self._check_grasp(gripper=self.robots[0].gripper,
                                            object_geoms=[g for g in obj.contact_geoms]))
            return grasped

        @sensor(modality=modality)
        def object_touched(obs_cache):
            touched = int(self.check_contact(self.robots[0].gripper, obj))
            return touched

        @sensor(modality=modality)
        def obj_to_eef_pos(obs_cache):
            # Immediately return default value if cache is empty
            if any([name not in obs_cache for name in
                    [f"{obj_name}_pos", f"{obj_name}_quat", "world_pose_in_gripper"]]):
                return np.zeros(3)
            obj_pose = T.pose2mat((obs_cache[f"{obj_name}_pos"], obs_cache[f"{obj_name}_quat"]))
            rel_pose = T.pose_in_A_to_pose_in_B(obj_pose, obs_cache["world_pose_in_gripper"])
            rel_pos, rel_quat = T.mat2pose(rel_pose)
            obs_cache[f"{obj_name}_to_{pf}eef_quat"] = rel_quat
            return rel_pos

        @sensor(modality=modality)
        def obj_to_eef_quat(obs_cache):
            return obs_cache[f"{obj_name}_to_{pf}eef_quat"] if \
                f"{obj_name}_to_{pf}eef_quat" in obs_cache else np.zeros(4)

        sensors = [obj_pos, obj_quat, obj_to_eef_pos, obj_to_eef_quat]
        names = [f"{obj_name}_pos", f"{obj_name}_quat", f"{obj_name}_to_{pf}eef_pos", f"{obj_name}_to_{pf}eef_quat"]

        if not obj_name in ["stove", "target"]:
            sensors += [obj_grasped, object_touched]
            names += [f"{obj_name}_grasped", f"{obj_name}_touched"]

        return sensors, names

    def _create_geom_sensors(self, obj_name, modality="object"):
        """
        Helper function to create sensors for a given object. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.

        Args:
            obj_name (str): Name of object to create sensors for
            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given obj
                names (list): array of corresponding observable names
        """

        if obj_name == "pot_handle":
            geom_id = self.pot_right_handle_id
        elif obj_name == "button_handle":
            geom_id = self.button_switch_pad_id
        else:
            raise NotImplementedError

        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality=modality)
        def obj_pos(obs_cache):
            return self.sim.data.geom_xpos[geom_id]

        @sensor(modality=modality)
        def obj_to_eef_pos(obs_cache):
            # Immediately return default value if cache is empty
            if any([name not in obs_cache for name in
                    [f"{obj_name}_pos", "world_pose_in_gripper"]]):
                return np.zeros(3)
            obj_pose = T.pose2mat((obs_cache[f"{obj_name}_pos"], np.array([1.0, 0.0, 0.0, 0.0])))
            rel_pose = T.pose_in_A_to_pose_in_B(obj_pose, obs_cache["world_pose_in_gripper"])
            rel_pos, _ = T.mat2pose(rel_pose)
            return rel_pos

        sensors = [obj_pos, obj_to_eef_pos]
        names = [f"{obj_name}_pos", f"{obj_name}_to_{pf}eef_pos"]

        return sensors, names

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 1.0 is provided if the task succeeds.

        Args:
            action (np.array): [NOT USED]

        Returns:
            float: reward value
        """

        meatball_in_pot = self.check_contact(self.objects_dict["meatball"], "pot_body_bottom")
        butter_in_pot = self.check_contact(self.objects_dict["butter"], "pot_body_bottom")
        pot_on_stove = self.check_contact("stove_collision_burner", "pot_body_bottom")

        pot_pos = self.sim.data.body_xpos[self.obj_body_id["pot"]]

        # TODO: the privileged info logic might need change
        if self.place_on_red:
            target_id = self.serving_region_red_body_id # self.obj_body_id[target_name]
        else:
            target_id = self.serving_region_green_body_id
        target_pos = self.sim.data.body_xpos[target_id]
        target_pot_xy_dist = np.linalg.norm(pot_pos[:2] - target_pos[:2])

        pot_touched = int(self.check_contact(self.robots[0].gripper, self.objects_dict["pot"]))

        pot_on_target = target_pot_xy_dist < 0.07 and not pot_touched

        if self.pick_bread:
            target_status = self.butter_melt_status
            target_in_pot = butter_in_pot
        else:
            target_status = self.meatball_cook_status
            target_in_pot = meatball_in_pot

        # This won't be the exact cook, since toggle takes time
        if self.cook_time == 0:
            aim_status = 0.7
        else:
            aim_status = 0.2

        if target_status < aim_status:
            # self.stage = int(meatball_in_pot) + int(self.butter_melt_status == 1) + int(pot_on_stove) + int(self.button_on)
            if not target_in_pot:
                stage = 0
            elif not pot_on_stove:
                stage = 1
            elif not self.button_on:
                stage = 2
            else:
                stage = 3 # we would do nothing in this stage
        else:
            if self.button_on:
                stage = 4
            else:
                stage = 5
            # self.stage = 5 + int(pot_on_target) + int(not self.button_on)
        self.stage = stage
        return int(self._check_success())

    def _check_success(self):
        pot_pos = self.sim.data.body_xpos[self.obj_body_id["pot"]]

        region_pos = self.sim.data.body_xpos[self.serving_region_red_body_id] if self.place_on_red else \
            self.sim.data.body_xpos[self.serving_region_green_body_id]

        dist_to_region = np.abs(pot_pos - region_pos)

        contact = self.check_contact(self.objects_dict['pot'], self.serving_region_red) if self.place_on_red else \
            self.check_contact(self.objects_dict['pot'], self.serving_region_green)
        
        obj_cooked = self.butter_melt_status > 0 if self.pick_bread else self.meatball_cook_status > 0
        obj_in_pot = self.butter_in_pot if self.pick_bread else self.meatball_in_pot
        return np.all(dist_to_region < 0.05) and contact and obj_cooked and self.correct_cook_wait_action and obj_in_pot

    def _check_failure(self):
        return False

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
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.food_menu)

    def _check_grasp_tolerant(self, gripper, object_geoms):
        """
        Tolerant version of check grasp function - often needed for checking grasp with Shapenet mugs.

        """
        check_1 = self._check_grasp(gripper=gripper, object_geoms=object_geoms)

        check_2 = self._check_grasp(gripper=["gripper0_finger1_collision", "gripper0_finger2_pad_collision"],
                                    object_geoms=object_geoms)

        check_3 = self._check_grasp(gripper=["gripper0_finger2_collision", "gripper0_finger1_pad_collision"],
                                    object_geoms=object_geoms)

        return check_1 or check_2 or check_3

    def get_processed_obs(self):
        obs = self._get_observations(force_update=True)
        for k in obs.keys():
            if 'image' in k:
                obs[k] = np.flip(cv2.resize(cv2.cvtColor(obs[k], cv2.COLOR_RGB2BGR).astype(float), (84, 84)), 0)
        obs = self.append_extra_obs(obs)
        for key in ['meatball_cook_status', 'butter_melt_status']:
            obs[key] = np.expand_dims(obs[key], 0) # honestly I don't know why we have to do this

        del obs['agentview_image']
        return obs

    def append_extra_obs(self, obs):
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

    def step(self, action):
        # update cooking status
        increase_rate = 0.03
        self.total_env_steps_taken += 1

        if self.button_on and self.pot_on_stove:
            if self.butter_in_pot:
                # TODO: the buffer never disappears
                # prev_butter_melt_status = self.butter_melt_status
                self.butter_melt_status = min(self.butter_melt_status + increase_rate, 1)
                # if prev_butter_melt_status < 1 and self.butter_melt_status == 1:
                #     butter_obj = self.objects_dict["butter"]
                #     body_id = self.obj_body_id["butter"]
                #     self.sim.data.set_joint_qpos(butter_obj.joints[0], np.concatenate([np.array((0, 0, 0)), self.sim.model.body_quat[body_id]]))
            if self.meatball_in_pot:
                # if self.butter_melt_status != 1:
                #     self.meatball_overcooked = True
                # elif not self.meatball_overcooked:
                self.meatball_cook_status = min(self.meatball_cook_status + increase_rate, 1)

        obs, reward, done, info = super().step(action)
        obs = self.get_processed_obs()
        self.record_obs()
        return obs, reward, self._check_success() or done, info

    def reset(self):
        self.stage = 0
        self.stage_id = 0
        self.total_env_steps_taken = 0
        self.correct_cook_wait_action = False
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

        # This only needs to be called once since the information do not change
        if self.pi_format == "clip":
            pi_text = self.generate_pi_text()
            self.current_text_embeds = self.get_text_embeddings(pi_text)

        return obs


class OracleKitchen(KitchenEnv, OracleBase):

    def __init__(self, **kwargs):
        KitchenEnv.__init__(self, **kwargs)
        OracleBase.__init__(self, self.action_spec)

    def reset_skills(self):
        super().reset_skills()

        # set the first skill to start
        # self.cur_skill = 'pick_cube'
        self.stage = 0

        # from kitchen_skills import ToggleSkill
        # # Test toggle button
        # self.toggle_on_skill = ToggleSkill(True)
        # self.toggle_off_skill = ToggleSkill(False)
        self.toggle_done = False

        self.update_task_progress()

    def reset(self):
        obs = super().reset()
        self.reset_skills()

        return obs

    def create_skills(self):
        self.skills = {
            'meatball_to_pot': [
                ['move', 'meatball_pos', np.array([0, 0, 0.05])],
                ['move', 'meatball_pos', np.array([0, 0, 0])],
                ['grasp', [0, 0, 0, 1], 5],
                ['move', 'delta', np.array([0, 0, 0.1])],
                ['move', 'pot_pos', np.array([0, 0, 0.12])],
                ['grasp', [0, 0, 0, -1], 5],
            ],

            'butter_to_pot': [
                ['move', 'butter_pos', np.array([0, 0, 0.05])],
                ['move', 'butter_pos', np.array([0, 0, 0])],
                ['grasp', [0, 0, 0, 1], 5],
                ['move', 'delta', np.array([0, 0, 0.1])],
                ['move', 'pot_pos', np.array([0, 0, 0.12])],
                ['grasp', [0, 0, 0, -1], 5],
            ],

            'pot_to_stove': [
                ['move', 'pot_handle_pos', np.array([0, 0, 0.05])],
                ['move', 'pot_handle_pos', np.array([0, 0, -0.01])],
                ['grasp', [0, 0, 0, 1], 5],
                ['move', 'delta', np.array([0, 0, 0.11])],
                ['move', 'stove_pos', np.array([0, -0.06, 0.12])],
                ['grasp', [0, 0, 0, -1], 5],
                ['move', 'delta', np.array([0, 0, 0.1])],
            ],

            'toggle_button': [
                ['move', 'button_pos', np.array([0, -0.02, 0.15])],
                ['toggle_on', 'button'],
                ['move', 'delta', np.array([0, 0, 0.1])],
                ['grasp', [0, 0, 0, -1], 5],
            ],

            'no_op': [
                ['no_op']
            ],

            'toggle_off': [
                ['move', 'button_pos', np.array([0, 0.02, 0.15])],
                ['toggle_off', 'button'],
                ['move', 'delta', np.array([0, 0, 0.1])],
                ['grasp', [0, 0, 0, -1], 5],
            ],

            'pot_to_red': [
                ['move', 'pot_handle_pos', np.array([0, 0, 0.11])],
                ['move', 'pot_handle_pos', np.array([0, 0, -0.01])],
                ['grasp', [0, 0, 0, 1], 5],
                ['move', 'delta', np.array([0, 0, 0.12])],
                ['move', 'serving_region_red_pos', np.array([0, -0.06, 0.12])],
                ['grasp', [0, 0, 0, -1], 5],
                ['move', 'delta', np.array([0, 0, 0.1])],
            ],

            'pot_to_green': [
                ['move', 'pot_handle_pos', np.array([0, 0, 0.11])],
                ['move', 'pot_handle_pos', np.array([0, 0, -0.01])],
                ['grasp', [0, 0, 0, 1], 5],
                ['move', 'delta', np.array([0, 0, 0.12])],
                ['move', 'serving_region_green_pos', np.array([0, -0.06, 0.12])],
                ['grasp', [0, 0, 0, -1], 5],
                ['move', 'delta', np.array([0, 0, 0.1])],
            ],
        }
        self.skills = OrderedDict(self.skills)



    def update_task_progress(self):
        # We can either use stage or use the ...
        if self.stage == 0:
            if self.pick_bread:
                self.set_task('butter_to_pot')
            else:
                self.set_task('meatball_to_pot')
        elif self.stage == 1:
            self.set_task('pot_to_stove')
        elif self.stage == 2:
            self.set_task('toggle_button')
        elif self.stage == 3: # self.cook_time determines how many steps of no_op will we do
            self.set_task('no_op')
        elif self.stage == 4:
            self.set_task('toggle_off')
        elif self.stage == 5:
            if self.place_on_red:
                self.set_task('pot_to_red')
            else:
                self.set_task('pot_to_green')


if __name__ == "__main__":
    from l2l.envs.robosuite.kitchen import KitchenEnv, OracleKitchen
    import robosuite as suite
    from robosuite.controllers import load_controller_config
    import cv2
    from l2l.config.env.robosuite.kitchen import env_config
    from l2l.wrappers.robosuite_wrappers import RobosuiteGymWrapper, EncoderRewardWrapper


    # initialize the task
    # env = suite.make(
    #     env_name="OracleKitchen",
    #     robots="Panda",
    #     controller_configs=load_controller_config(default_controller="OSC_POSITION"),
    #     has_renderer=True,
    #     has_offscreen_renderer=True,
    #     ignore_done=True,
    #     use_camera_obs=True,
    #     control_freq=10,
    #     camera_names=["agentview", "sideview", "robot0_eye_in_hand"],
    # )

    env = RobosuiteGymWrapper(suite.make(**env_config))
    obs, _ = env.reset()

    # Get action limits
    low, high = env.action_spec

    # do visualization
    for i in range(10000):
        action = np.random.normal(0, 0.1, 4) + env.get_optimal_action(obs)
        # action = np.zeros_like(action)
        obs, reward, done, _, _ = env.step(action)

        # img = np.concatenate([obs['agentview_image'], obs['sideview_image'], obs['robot0_eye_in_hand_image']],
        #                      axis=1).astype(np.uint8)
        # cv2.imshow('img', img)
        # cv2.waitKey(5)

        move_cam = False
        if move_cam:
            for i in range(1):
                camera_action = np.random.choice([0, 1, 2])
                obs, _, _, _ = env.unwrapped.rotate_camera(camera_action)

        img = obs['activeview_image'].astype(np.float32)/255
        cv2.imshow('img', img)
        cv2.waitKey(5)

        env.render()

        if done:
            obs, _ = env.reset()