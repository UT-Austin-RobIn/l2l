import h5py
import copy
from l2l.utils.general_utils import recursive_dict_write_in_grp

class BaseDataGenerator:

    def make_dataset(self, num_traj, episode_length=None, noise=0, save_path=None, render=False, *args, **kwargs):
        if save_path:
            f = h5py.File(save_path, 'w')
            grp = f.create_group('data')

        total_num_samples = 0
        i = 0
        while i < num_traj:
            trajectory = self.generate_trajectory(episode_length=episode_length, noise=noise, render=render)
            assert 'actions' in trajectory.keys(), 'Action key missing from demo'

            # only save successful trajectories
            if trajectory['dones'][-1] != 1:
                continue
            print(f"Successfully generated trajectory {i} with length {len(trajectory['dones'])}")

            total_num_samples += trajectory['actions'].shape[0]
            if save_path:
                ep_grp = grp.create_group(f'demo_{i}')
                recursive_dict_write_in_grp(ep_grp, data=trajectory)
                
                ep_grp.attrs["num_samples"] = trajectory['actions'].shape[0]

            i += 1

        if save_path:
            grp.attrs['num_demos'] = num_traj
            grp.attrs['total'] = total_num_samples

            # some extra info for robomimic
            try:
                import json
                env_config = copy.deepcopy(self.env_config)
                env_name = env_config.env_name
                del env_config['env_name']
                grp.attrs['env_args'] = json.dumps(dict(env_name=env_name, env_kwargs=env_config, type=1), indent=4)
            except:
                pass

            f.close()
    
    def generate_trajectory(self, episode_length=None, render=False):
        raise NotImplementedError