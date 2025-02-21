from robosuite.controllers import load_controller_config
from l2l.envs.robosuite.walled_multi_stage import OracleWalledMultiStage
from l2l.utils.general_utils import AttrDict

env_config = AttrDict(
        env_name= "OracleWalledMultiStage",
        robots= "Panda",
        controller_configs=load_controller_config(default_controller="OSC_POSITION"),
        has_renderer=True,
        has_offscreen_renderer=True,
        camera_names=['activeview', 'robot0_eye_in_hand', 'agentview'],
        ignore_done=False,
        horizon=150,
        use_camera_obs=True,
        control_freq=10,
        pi_format="onehot", # "clip", "onehot", "continuous"
        partitioned_encoding=False, # whether to have separate encoding for each sentence
        text_encoder=None,
)