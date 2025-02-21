import numpy as np

class OracleBase:

    def __init__(self, action_spec) -> None:
        self.low, self.high = action_spec

        self.create_skills()
        self.reset_skills()

    def reset_skills(self):
        self.goal_idx = 0
        self.grasp_counter = 0
        self.goal_updated = False
        self.to_xyz = None
        self.from_xyz = None
        self.last_gripper_act = -1

    def move(self, from_xyz, to_xyz):
        """Computes action components that help move from 1 position to another.

        Args:
            from_xyz (np.ndarray): The coordinates to move from (usually current position)
            to_xyz (np.ndarray): The coordinates to move to
            p (float): constant to scale response

        Returns:
            (np.ndarray): Response that will decrease abs(to_xyz - from_xyz)

        """
        error = (to_xyz - from_xyz)
        response = 0.5 * error/np.linalg.norm(error)
        response = np.clip(response, self.low[:3], self.high[:3])
        # print(to_xyz, from_xyz, response)
        return response

    def update_goal_on_completion(self, obs):
        current_goal = self.skills[self.cur_skill][self.goal_idx]
        if current_goal[0] == 'move':
            if np.linalg.norm(self.from_xyz - self.to_xyz) < 0.02: # threshold distance for reaching
                if self.goal_idx < len(self.skills[self.cur_skill])-1:
                    self.grasp_counter = 0
                    self.goal_idx += 1
                    self.goal_updated = True
                else:
                    self.update_task_progress()
            else:
                self.goal_updated = False

        elif current_goal[0] == 'grasp':
            if self.grasp_counter >= current_goal[2]:
                if self.goal_idx < len(self.skills[self.cur_skill])-1:
                    self.grasp_counter = 0
                    self.goal_idx += 1
                    self.goal_updated = True
                else:
                    self.update_task_progress()
            else:
                self.grasp_counter += 1
                self.goal_updated = False

        elif current_goal[0] in ['toggle_on', "toggle_off"]:
            if self.toggle_done:
                if self.goal_idx < len(self.skills[self.cur_skill])-1:
                    self.goal_idx += 1
                    self.goal_updated = True
                else:
                    self.update_task_progress()
                self.toggle_done = False
            else:
                self.goal_updated = False

        elif current_goal[0] == 'no_op':
            if self.goal_idx < len(self.skills[self.cur_skill])-1:
                self.goal_idx += 1
                self.goal_updated = True
            else:
                self.update_task_progress()

        else:
            raise NotImplementedError
            
    def get_goal_pos(self, obs):
        current_goal = self.skills[self.cur_skill][self.goal_idx]
        to_xyz = self.to_xyz
        if current_goal[0] == 'move':
            if current_goal[1] == 'delta':
                if self.goal_updated:
                    to_xyz = self.from_xyz + current_goal[2]
            else:
                to_xyz = obs[current_goal[1]] + current_goal[2]

        return to_xyz
    
    def set_task(self, task_id):
        self.cur_skill = task_id
        self.goal_idx = 0
        self.goal_updated = True

    def get_optimal_action(self, obs, noise = 0):
        # print(self.cur_skill, self.skills[self.cur_skill][self.goal_idx])
        current_goal = self.skills[self.cur_skill][self.goal_idx]
        self.from_xyz = obs['robot0_eef_pos']

        action = np.zeros(4)
        action[3] = self.last_gripper_act
        self.to_xyz = self.get_goal_pos(obs)
        self.goal_updated = False

        if current_goal[0] == 'move':
            action[:3] = self.move(self.from_xyz, self.to_xyz) + np.random.normal(0, noise, 3)
        elif current_goal[0] == 'grasp':
            action = current_goal[1]
        elif current_goal[0] == 'toggle_on':
            action, self.toggle_done = self.toggle_on_skill.step(obs)
        elif current_goal[0] == 'toggle_off':
            action, self.toggle_done = self.toggle_off_skill.step(obs)
        elif current_goal[0] == 'no_op':
            pass
        else:
            raise NotImplementedError
        
        self.last_gripper_act = action[-1]
        self.update_goal_on_completion(obs)
        return action
    
    def create_skills(self):
        raise NotImplementedError
    
    def update_task_progress(self):
        raise NotImplementedError