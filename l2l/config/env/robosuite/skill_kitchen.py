from robosuite.controllers import load_controller_config
from l2l.envs.robosuite.skill_kitchen import SkillKitchenEnv
from l2l.utils.general_utils import AttrDict

from transformers import AutoTokenizer, CLIPTextModelWithProjection

model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")


env_config = AttrDict(
        env_name= "SkillKitchenEnv",
        robots= "Panda",
        controller_configs=load_controller_config(default_controller="OSC_POSITION"),
        has_renderer=True,
        has_offscreen_renderer=True,
        camera_names=['activeview', 'robot0_eye_in_hand', 'agentview'],
        ignore_done=True,
        use_camera_obs=True,
        control_freq=10,
        horizon=15,
        pi_format="onehot", # "clip"
        partitioned_encoding=False, # whether to have separate encoding for each sentence
        text_encoder=[tokenizer, model],
)