"""
modified from
https://github.com/ARISE-Initiative/robosuite/blob/maple/robosuite/controllers/skill_controller.py
and
https://github.com/ARISE-Initiative/robosuite/blob/maple/robosuite/controllers/skills.py
"""

import numpy as np


class BaseSkill:
    def __init__(self, num_max_steps=50):
        self.reach_threshold = 0.02
        self.grasp_reach_threshold = 0.005
        self.lift_height = 0.95
        self.pot_lift_height = 0.90
        self.controller_scale = 0.05
        self.button_push_offset = 0.08
        self.eef_blocked_threshold = 1e-3

        self.num_max_steps = num_max_steps
        self.num_step = 0

        self.prev_eef_pos = None
        self.num_blocked_steps = 0
        self.blocked = False

    def update_blocked_status(self, obs):
        eef_pos = obs["robot0_eef_pos"]
        if self.prev_eef_pos is None or np.linalg.norm(eef_pos - self.prev_eef_pos) > self.eef_blocked_threshold:
            self.num_blocked_steps = 0
        else:
            self.num_blocked_steps += 1

        if self.num_blocked_steps >= 3:
            self.blocked = True

        self.prev_eef_pos = eef_pos

    def get_delta_pos(self, eef, goal_pos):
        return np.clip((goal_pos - eef) / self.controller_scale, -1, 1)

    def update_state(self, obs):
        pass

    def get_pos_ac(self, obs):
        raise NotImplementedError

    def get_gripper_ac(self, obs):
        raise NotImplementedError

    def is_success(self, obs):
        return False

    def step(self, obs):
        self.update_state(obs)
        pos = self.get_pos_ac(obs)
        grp = self.get_gripper_ac(obs)
        done = (self.num_step >= self.num_max_steps) or self.is_success(obs) or self.blocked
        self.num_step += 1
        return np.concatenate([pos, grp]), done


class DropSkill(BaseSkill):
    def __init__(self, in_hand_obj_touch_key):
        super().__init__()
        self.in_hand_obj_touch_key = in_hand_obj_touch_key
        self.num_no_touch_steps = 0

    def update_state(self, obs):
        if self.in_hand_obj_touch_key is not None:
            if obs[self.in_hand_obj_touch_key]:
                self.num_no_touch_steps = 0
            else:
                self.num_no_touch_steps += 1

    def get_pos_ac(self, obs):
        return np.array([0, 0, 0.05])

    def get_gripper_ac(self, obs):
        return np.ones(1) * -1

    def is_success(self, obs):
        if self.in_hand_obj_touch_key is None:
            return True

        return self.num_no_touch_steps >= 5


class MoveSkill(BaseSkill):
    def __init__(self, in_hand_obj_pos_key, target_pos_key, z_offset, thing_in_hand):
        super().__init__()
        self.in_hand_obj_pos_key = in_hand_obj_pos_key
        self.target_pos_key = target_pos_key
        self.z_offset = z_offset
        self.thing_in_hand = thing_in_hand

        self.num_reached_steps = 0

        self.state = "LIFTING"
        self.STATES = ["LIFTING", "HOVERING", "LOWERING"]

    def get_target_pos(self, obs):
        if isinstance(self.target_pos_key, str):
            target_pos = np.copy(obs[self.target_pos_key])
            if self.target_pos_key == "button_pos":
                target_pos[2] = obs["button_handle_pos"][2]
            target_pos[2] += self.z_offset
            return target_pos
        else:
            return self.target_pos_key

    def update_state(self, obs):
        mov_pos = obs[self.in_hand_obj_pos_key]
        target_pos = self.get_target_pos(obs)

        th = self.reach_threshold
        if self.in_hand_obj_pos_key == "pot_pos":
            lift_height = self.pot_lift_height
        else:
            lift_height = self.lift_height
        reached_lift = (mov_pos[2] >= lift_height - th)
        reached_xy = (np.linalg.norm(mov_pos[0:2] - target_pos[0:2]) < th)
        reached_xyz = (np.linalg.norm(mov_pos - target_pos) < th)

        if self.state == "LOWERING" or reached_xy:
            self.state = "LOWERING"
            if reached_xyz:
                self.num_reached_steps += 1
            else:
                self.num_reached_steps = 0
        elif self.state == "HOVERING" or (self.state == "LIFTING" and reached_lift):
            self.state = "HOVERING"
        else:
            self.state = "LIFTING"

        if self.state == "LOWERING":
            self.update_blocked_status(obs)

        assert self.state in self.STATES

    def get_pos_ac(self, obs):
        mov_pos = obs[self.in_hand_obj_pos_key]
        target_pos = self.get_target_pos(obs)

        if self.state == "LIFTING":
            goal_pos = mov_pos
        else:
            goal_pos = target_pos
        goal_pos = np.copy(goal_pos)

        if self.state == "LOWERING":
            goal_pos[2] = target_pos[2]
        else:
            if self.in_hand_obj_pos_key == "pot_pos":
                lift_height = self.pot_lift_height
            else:
                lift_height = self.lift_height
            goal_pos[2] = lift_height

        pos = self.get_delta_pos(mov_pos, goal_pos)

        return pos

    def get_gripper_ac(self, obs):
        if self.thing_in_hand:
            return np.ones(1)
        else:
            return np.ones(1) * -1

    def is_success(self, obs):
        return self.num_reached_steps >= 3


class GraspSkill(BaseSkill):
    def __init__(self, obj_pos_key, grasped_key):
        super().__init__()
        self.obj_pos_key = obj_pos_key
        self.grasped_key = grasped_key

        self.state = "REACHING"
        self.STATES = ["REACHING", "GRASPING"]
        self.num_reached_steps = 0
        self.num_grasped_steps = 0

    def update_state(self, obs):
        eef_pos = obs["robot0_eef_pos"]
        obj_pos = obs[self.obj_pos_key]

        if self.obj_pos_key == "meatball_pos":
            reach_th = self.grasp_reach_threshold
        else:
            reach_th = self.reach_threshold
        reached_xyz = np.linalg.norm(eef_pos - obj_pos) < reach_th
        grasped = obs[self.grasped_key]

        if self.state == "GRASPING" or (self.state == "REACHING" and self.num_reached_steps >= 5):
            self.state = "GRASPING"
            if grasped:
                self.num_grasped_steps += 1
        else:
            self.state = "REACHING"
            if reached_xyz:
                self.num_reached_steps += 1

        assert self.state in self.STATES

    def get_pos_ac(self, obs):
        eef_pos = obs["robot0_eef_pos"]
        obj_pos = obs[self.obj_pos_key]

        pos = self.get_delta_pos(eef_pos, obj_pos)
        return pos

    def get_gripper_ac(self, obs):
        if self.state == "GRASPING":
            return np.ones(1)
        else:
            return np.ones(1) * -1

    def is_success(self, obs):
        return self.num_grasped_steps >= 5


class ToggleSkill(BaseSkill):
    def __init__(self, target_button_state):
        super().__init__()
        self.target_button_state = target_button_state

        self.state = "HOVERING"
        self.STATES = ["HOVERING", "LOWERING", "PUSHING"]

    def update_state(self, obs):
        eef_pos = obs["robot0_eef_pos"]
        start_pos = np.copy(obs["button_handle_pos"])

        if self.target_button_state:
            start_pos += np.array([0, -self.button_push_offset, 0])
        else:
            start_pos += np.array([0, self.button_push_offset, 0])

        th = self.reach_threshold
        reached_xy = (np.linalg.norm(eef_pos[0:2] - start_pos[0:2]) < th)
        reached_xyz = (np.linalg.norm(eef_pos - start_pos) < th)

        if self.state == "PUSHING" or (self.state == "LOWERING" and reached_xyz):
            self.state = "PUSHING"
        elif self.state == "LOWERING" or (self.state == "HOVERING" and reached_xy):
            self.state = "LOWERING"
        else:
            self.state = "HOVERING"

        assert self.state in self.STATES

    def get_pos_ac(self, obs):
        eef_pos = obs["robot0_eef_pos"]
        button_handle_pos = obs["button_handle_pos"]

        start_pos = np.copy(button_handle_pos)
        end_pos = np.copy(button_handle_pos)
        if self.target_button_state:
            start_pos += np.array([0, -self.button_push_offset, 0])
            end_pos += np.array([0, self.button_push_offset, 0])
        else:
            start_pos += np.array([0, self.button_push_offset, 0])
            end_pos += np.array([0, -self.button_push_offset, 0])

        if self.state in ["HOVERING", "LOWERING"]:
            goal_pos = np.copy(start_pos)
        else:
            goal_pos = np.copy(end_pos)

        if self.state == "HOVERING":
            goal_pos[2] = button_handle_pos[2] + 2 * self.controller_scale

        pos = self.get_delta_pos(eef_pos, goal_pos)
        return pos

    def get_gripper_ac(self, obs):
        return np.ones(1)

    def is_success(self, obs):
        if obs["button_joint_qpos"] >= self.button_push_offset:
            button_on = True
        elif obs["button_joint_qpos"] < -self.button_push_offset:
            button_on = False
        else:
            button_on = None
        return button_on == self.target_button_state


