from tiago_gym import TiagoGym
from l2l.utils.general_utils import AttrDict

from transformers import AutoTokenizer, CLIPTextModelWithProjection

model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

env_config = AttrDict(
    env_class=TiagoGym,
    env_kwargs=AttrDict(
        frequency=1,
        head_enabled=True,
        base_enabled=False,
        torso_enabled=False,
        right_arm_enabled=False,
        left_arm_enabled=False,
        right_gripper_type='robotiq2F-140',
        left_gripper_type='robotiq2F-85',
    ),
    pi_format="onehot", # "clip", "onehot", "continuous"
    # partitioned_encoding=True, # whether to have separate encoding for each sentence
    # text_encoder=[tokenizer, model],
)