from robosuite.controllers import load_controller_config
from l2l.envs.robosuite.two_arm import TwoArmOracle
from l2l.utils.general_utils import AttrDict

env_config = AttrDict(
        env_name= "TwoArmOracle",
        robots= ["Panda","Panda"],
        controller_configs=[load_controller_config(default_controller="OSC_POSITION"), load_controller_config(default_controller="OSC_POSITION")],
        has_renderer=True,
        has_offscreen_renderer=True,
        camera_names=['activeview', 'robot0_eye_in_hand', 'agentview'],
        ignore_done=False,
        horizon=150,
        use_camera_obs=True,
        control_freq=10,
)
