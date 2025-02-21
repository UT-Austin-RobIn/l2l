import cv2
import numpy as np
from collections import OrderedDict

from gymnasium import spaces

from l2l.envs.robosuite.kitchen import KitchenEnv
from l2l.envs.robosuite.skill_base import SkillBase
from l2l.envs.robosuite.kitchen_skills import ToggleSkill


class SkillKitchenEnv(KitchenEnv, SkillBase):

	def __init__(self, **kwargs):
		KitchenEnv.__init__(self, **kwargs)
		SkillBase.__init__(self, self.action_spec)

	def reset(self):
		obs = super().reset()
		self.reset_skills()

		self.toggle_on_skill = ToggleSkill(True)
		self.toggle_off_skill = ToggleSkill(False)

		return obs

	@property
	def action_space(self):
		return spaces.MultiDiscrete([len(self.skills)])

	def create_skills(self):
		self.skills = {
			'meatball_to_pot': [
				['move', 'meatball_pos', np.array([0, 0, 0.05])],
				['move', 'meatball_pos', np.array([0, 0, 0.0])],
				['grasp', [0, 0, 0, 1], 10],
				['move', 'delta', np.array([0, 0, 0.1])],
				['move', 'pot_pos', np.array([0, 0, 0.12])],
				['grasp', [0, 0, 0, -1], 10],
			],

			'butter_to_pot': [
				['move', 'butter_pos', np.array([0, 0, 0.05])],
				['move', 'butter_pos', np.array([0, 0, 0.0])],
				['grasp', [0, 0, 0, 1], 10],
				['move', 'delta', np.array([0, 0, 0.1])],
				['move', 'pot_pos', np.array([0, 0, 0.12])],
				['grasp', [0, 0, 0, -1], 10],
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

			'cook_five': [
				['move', 'button_pos', np.array([0, -0.02, 0.15])],
				['toggle_on', 'button'],
				['move', 'delta', np.array([0, 0, 0.1])],
				['grasp', [0, 0, 0, -1], 5],
				["wait", None, 5],
				['move', 'button_pos', np.array([0, 0.02, 0.15])],
				['toggle_off', 'button'],
				['move', 'delta', np.array([0, 0, 0.1])],
				['grasp', [0, 0, 0, -1], 5],
			],

			'cook_one': [
				['move', 'button_pos', np.array([0, -0.02, 0.15])],
				['toggle_on', 'button'],
				['move', 'delta', np.array([0, 0, 0.1])],
				['grasp', [0, 0, 0, -1], 5],
				["wait", None, 1],
				['move', 'button_pos', np.array([0, 0.02, 0.15])],
				['toggle_off', 'button'],
				['move', 'delta', np.array([0, 0, 0.1])],
				['grasp', [0, 0, 0, -1], 5],
			],

			'pot_to_red': [
				['move', 'pot_handle_pos', np.array([0, 0, 0.11])],
				['move', 'pot_handle_pos', np.array([0, 0, -0.01])],
				['grasp', [0, 0, 0, 1], 5],
				['move', 'delta', np.array([0, 0, 0.10])],
				['move', 'serving_region_red_pos', np.array([0, -0.06, 0.12])],
				['grasp', [0, 0, 0, -1], 5],
				['move', 'delta', np.array([0, 0, 0.1])],
			],

			'pot_to_green': [
				['move', 'pot_handle_pos', np.array([0, 0, 0.11])],
				['move', 'pot_handle_pos', np.array([0, 0, -0.01])],
				['grasp', [0, 0, 0, 1], 5],
				['move', 'delta', np.array([0, 0, 0.10])],
				['move', 'serving_region_green_pos', np.array([0, -0.06, 0.12])],
				['grasp', [0, 0, 0, -1], 5],
				['move', 'delta', np.array([0, 0, 0.1])],
			],
		}
		self.skills = OrderedDict(self.skills)

	def get_optimal_skill_sequence(self):

		stops = 'cook_one' if self.cook_time == 0 else 'cook_five'
		place_pot = 'pot_to_red' if self.place_on_red else 'pot_to_green'
		cook_item = 'butter_to_pot' if self.pick_bread else 'meatball_to_pot'

		return [cook_item, 'pot_to_stove', stops, place_pot]

	def step(self, skill_id):
		skill_id = int(skill_id)
		assert len(self.skills) > skill_id >= 0, f"Invalid skill_id: {skill_id}"

		self.skill_step += 1
		skill_name = self.id_to_skill[skill_id]

		item_in_pot = self.butter_in_pot if self.pick_bread else self.meatball_in_pot
		if item_in_pot and self.pot_on_stove:
			if self.cook_time == 0: 
				if skill_name == 'cook_one':
					self.correct_cook_wait_action = True
			else:
				if skill_name == 'cook_five':
					self.correct_cook_wait_action = True
			
		return self.execute_skill(skill_name)

	def execute_skill(self, skill_name):
		current_skill = self.skills[skill_name]

		render = False
		# print(f"start executing skill {skill_name}")

		obs = self.get_processed_obs()
		reward = 0
		done = False
		info = {}

		for skill in current_skill:
			if skill[0] == 'move':
				from_xyz = obs['robot0_eef_pos']
				to_xyz = self.get_goal_pos(obs, skill)

				max_loops = 40
				while max_loops > 0 and np.linalg.norm(from_xyz - to_xyz) > 0.01: # 0.02:
					action = np.zeros(4)
					action[:3] = self.move(from_xyz, to_xyz)
					action[3] = self.last_gripper_act
					obs, reward, done, info = super().step(action)

					# TODO: this is just for visualization
					if render:
						self.render()

					from_xyz = obs['robot0_eef_pos']

					max_loops -= 1

			elif skill[0] == 'grasp':
				n_grasp_steps = skill[2]
				for _ in range(max(1, n_grasp_steps)):
					action = skill[1]
					self.last_gripper_act = action[-1]
					obs, reward, done, info = super().step(action)

					if render:
						self.render()

			elif skill[0] == 'wait':
				action = np.zeros(4)
				action[3] = self.last_gripper_act
				for _ in range(skill[2]):
					obs, reward, done, info = super().step(action)

					if render:
						self.render()

			elif skill[0] == 'toggle_on':
				toggle_done = False
				max_loops = 40
				while max_loops > 0 and (not toggle_done):
					action, toggle_done = self.toggle_on_skill.step(obs)
					obs, reward, done, info = super().step(action)

					if render:
						self.render()
					max_loops -= 1

			elif skill[0] == 'toggle_off':
				toggle_done = False
				max_loops = 40
				while max_loops > 0 and (not toggle_done):
					action, toggle_done = self.toggle_off_skill.step(obs)
					obs, reward, done, info = super().step(action)

					if render:
						self.render()
					max_loops -= 1

			else:
				raise NotImplementedError
			if done:
				break

		return obs, reward, done or (self.skill_step > self.horizon), info


if __name__ == "__main__":
	from l2l.config.env.robosuite.skill_kitchen import env_config
	import robosuite as suite
	import cv2
	import matplotlib.pyplot as plt

	from l2l.wrappers.robosuite_wrappers import RobosuiteGymWrapper, UncertaintyBasedLookingWrapper
	from stable_baselines3 import PPO

	def get_encoder_path(rl_path):
		rl_ckpt = rl_path.split('/')[-3].split('_')[-1]

		encoder_path = "/".join(rl_path.split('/')[:-3]) + f"/encoder_weights/weights/weights_ep{rl_ckpt}.pth"
		return encoder_path


	test_cam_rl_policy = False

	if test_cam_rl_policy:
		from l2l.config.dual.robosuite.skill_kitchen.kitchen_action_dual_config import encoder_model_config

		# initialize the task
		# env = RobosuiteGymWrapper(suite.make(**env_config))
		env = UncertaintyBasedLookingWrapper(RobosuiteGymWrapper(suite.make(**env_config)))

		# rl_ckpt = '/home/shivin/Desktop/learning2look/experiments/dual_multi_stage_large_actions_no_stage_id_2/epoch_18/weights/rl_model_456800_steps.zip'
		# encoder_path = get_encoder_path(rl_ckpt)
		# cam_policy = PPO.load(rl_ckpt, env=env)
		# enc = encoder_model_config.model_class.load_weights(encoder_path)
		enc = encoder_model_config.model_class(encoder_model_config)

		env.set_encoder(enc)

		obs, _ = env.reset()

		action = 2
		for i in range(1000):
			action = 0#env.action_space.sample()
			# action, states = cam_policy.predict(
			# 	obs,  # type: ignore[arg-type]
			# 	deterministic=True,
			# )
			# if obs['camera_angle']>0.9 or obs['camera_angle']<-0.9:
			#     action = action%2 + 1
			#     print(action)

			# if i % 2 == 0:
			#     action = int(input("enter action:"))
			# print(i)

			obs, reward, done, _, _ = env.step(action)

			# for _ in range(10):
			#     loss = env.encoder.get_reward(obs)
			#     reward = np.clip(2*(0.5-loss), -1, 1)
			#     bins[int((obs['camera_angle'])*10)] += reward
			#     bin_counts[int((obs['camera_angle'])*10)] += 1
			# print(obs['camera_angle'], int((obs['camera_angle'] + 1)*100), reward)

			img = obs['activeview_image'].astype(np.float32) / 255
			# env.render()
			cv2.imshow('img', img)
			cv2.waitKey(1)

			# if obs["camera_angle"] < -0.95:
			if done:
				# print(env.uncertainty)
				# plt.plot(env.uncertainty)
				# plt.show()

				# print(bin_counts)
				# x = [bins[k]/bin_counts[k] if bin_counts[k] > 0 else 0 for k in bins.keys()]
				# plt.bar(list(bins.keys()), x)
				# plt.show()

				env.reset()

	else:
		from l2l.wrappers.robosuite_wrappers import FullRLBaselineWrapper
		env = FullRLBaselineWrapper(RobosuiteGymWrapper(suite.make(**env_config)))
		obs, _ = env.reset()

		steps = 0
		for i in range(10000):
			robot_action = env.get_optimal_action(obs)
			# robot_action = env.action_space.sample()
			obs, reward, done, _, _ = env.step(robot_action)

			# for i in range(10):
			# camera_action = np.random.choice([0, 1, 2, 3, 4])
			# # camera_action = int(input('enter camera action:'))
			# for _ in range(2):
			# 	obs, _, _, _ = env.unwrapped.rotate_camera(camera_action)

			steps += 1

			# print(obs['task_obs'], obs['privileged_info'])

			# print(obs['meatball_cook_status'], obs['butter_melt_status'])

			img = obs['activeview_image'].astype(np.float32)/255
			cv2.imshow('img', img)
			cv2.waitKey(5)
			print(obs['stage_id'])
			env.render()
			# input()
			print(reward)
			input()
			# if i % 20 == 0:
			#     camera_action = int(input())

			if done:
		
				input()
				env.reset()
				steps = 0