from robosuite.controllers import load_controller_config
from l2l.envs.robosuite.kitchen import OracleKitchen
from l2l.utils.general_utils import AttrDict

env_config = AttrDict(
        env_name= "OracleKitchen",
        robots= "Panda",
        controller_configs=load_controller_config(default_controller="OSC_POSITION"),
        has_renderer=True,
        has_offscreen_renderer=True,
        camera_names=['activeview', 'robot0_eye_in_hand', 'agentview', 'sideview'],
        ignore_done=False,
        horizon=500,
        use_camera_obs=True,
        control_freq=10,
)