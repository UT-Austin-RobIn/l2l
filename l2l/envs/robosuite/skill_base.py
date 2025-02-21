import numpy as np

class SkillBase:

    def __init__(self, action_spec) -> None:
        self.low, self.high = action_spec
        self.skill_step = 0

        self.create_skills()
        self.id_to_skill = {i: skill for i, skill in enumerate(self.skills.keys())}
        self.skill_to_id = {skill: i for i, skill in enumerate(self.skills.keys())}
    
    def reset_skills(self):
        self.cur_skill_idx = 0
        self.last_gripper_act = -1
        self.skill_step = 0
        self.optimal_skill_sequence = self.get_optimal_skill_sequence()

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
        # response = np.clip(response, self.low[:3], self.high[:3])
        return response
            
    def get_goal_pos(self, obs, skill, xyz_key='robot0_eef_pos'):
        from_xyz = obs[xyz_key]
        current_goal = skill
        if skill[0] == 'move':
            if skill[1] == 'delta':
                to_xyz = from_xyz + current_goal[2]
            else:
                to_xyz = obs[current_goal[1]] + current_goal[2]
        return to_xyz
    
    def set_task(self, task_id):
        self.cur_skill = task_id

    def get_optimal_action(self, obs, **kwargs):
        skill_id = self.skill_to_id[self.optimal_skill_sequence[self.cur_skill_idx]]
        self.cur_skill_idx = min(self.cur_skill_idx + 1, len(self.optimal_skill_sequence) - 1)
        return np.array([skill_id])

    def create_skills(self):
        raise NotImplementedError
    
    def get_optimal_skill_sequence(self):
        raise NotImplementedError
    
    def execute_skill(self, skill_id):
        raise NotImplementedError