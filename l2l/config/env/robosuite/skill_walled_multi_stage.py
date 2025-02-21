from robosuite.controllers import load_controller_config
from l2l.envs.robosuite.skill_walled_multi_stage import SkillWalledMultiStage
from l2l.utils.general_utils import AttrDict

from transformers import AutoTokenizer, CLIPTextModelWithProjection

model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")


env_config = AttrDict(
        env_name= "SkillWalledMultiStage",
        robots= "Panda",
        controller_configs=load_controller_config(default_controller="OSC_POSITION"),
        has_renderer=True,
        has_offscreen_renderer=True,
        camera_names=['activeview', 'robot0_eye_in_hand', 'agentview'],
        ignore_done=True,
        use_camera_obs=True,
        control_freq=10,
        horizon=10,
        pi_format="onehot", # "clip", "onehot", "continuous"
        partitioned_encoding=False, # whether to have separate encoding for each sentence
        text_encoder=[tokenizer, model],
)