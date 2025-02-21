import os
import numpy as np
import h5py
import cv2
import json
import pickle


def stats_for_actions_and_low_dim_obs(args, demos):
    action_list = []
    obs_list = {
        'right': [],
        'left': [],
        'base': [],
        'base_velocity': [],
    }
    for demo in demos:
        print(demo)
        with h5py.File(os.path.join(args.data_path, demo), 'r') as f:
            # action = np.concatenate((f['actions'][()][:, :9], f['actions'][()][:, 15:16]), axis=-1)
            action = f['actions'][()]
            action_list.append(action)

            for k in obs_list.keys():
                obs_list[k].append(f['obs'][k][:])
    
    stats = {
        'mean': {},
        'std': {},
        'max': {},
        'min': {}
    }

    action_list = np.concatenate(action_list, axis=0)
    stats['mean']['actions'] = np.mean(action_list, axis=0)
    stats['std']['actions'] = np.std(action_list, axis=0)
    stats['max']['actions'] = np.max(action_list, axis=0)
    stats['min']['actions'] = np.min(action_list, axis=0)

    for k in obs_list:
        obs_list[k] = np.concatenate(obs_list[k], axis=0)
        stats['mean'][k] = np.mean(obs_list[k], axis=0)
        stats['std'][k] = np.std(obs_list[k], axis=0)
        stats['max'][k] = np.max(obs_list[k], axis=0)
        stats['min'][k] = np.min(obs_list[k], axis=0)

    print(stats)
    return stats

def preprocess(data, stats):
    d = {}
    for k in data.keys():
        if k == 'obs':
            d[k] = {}
            for obs_key in data[k].keys():
                if ('image' in obs_key or 'depth' in obs_key):
                    obs_data = data[k][obs_key][:]
                    resized_obs_data = []
                    for img in obs_data:
                        if 'depth' in obs_key:
                            im = np.clip(im, 0, 4000)/4000
                            assert np.max(im) <= 1, print(np.max(img))
                            
                        resized_obs_data.append(cv2.resize(im.astype(float), (84, 84)))
                        
                    d[k][obs_key] = np.array(resized_obs_data)
                else:
                    if obs_key in stats['mean'].keys():
                        d[k][obs_key] = (data[k][obs_key][:] - stats['mean'][obs_key])/stats['std'][obs_key]
                    else:
                        d[k][obs_key] = data[k][obs_key][:]

        else:
            if k == 'actions':
                action = data[k][:]
                action[:, :-1] = (action[:, :-1] - stats['mean']['actions'][:-1])/stats['std']['actions'][:-1]
                d[k] = action
            else:
                d[k] = data[k][:]
    return d

def get_demo_id(filename):
    return int(filename.split('.')[0].split('_')[1])

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="path to data directory")
    parser.add_argument("--render", action="store_true", help="pass flag to render environment while data generation")
    parser.add_argument("--save_path", type=str, help="path to save directory")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    demos = sorted(os.listdir(args.data_path), key=get_demo_id)
    stats = stats_for_actions_and_low_dim_obs(args, demos) # generate data statistics

    # save data statistics to file
    dataset_stats_path = os.path.join(args.save_path, 'dataset_stats.pkl')
    with open(dataset_stats_path, 'wb') as f:
        pickle.dump(stats, f)
    
    # loop through each individual group
    n_outputs = int(np.ceil(len(demos)/args.group_size))
    for i in range(n_outputs):
        demo_low = i*args.group_size
        demo_high = min((i+1)*args.group_size, len(demos)) - 1

        data_path = os.path.join(args.save_path, f'data_{get_demo_id(demos[demo_low])}_to_{get_demo_id(demos[demo_high])}.h5')
        print('Writing to', data_path)

        with h5py.File(data_path, 'w') as data:
            grp = data.create_group('data')
            
            total_num_samples = 0
            total_num_demos = 0
            for demo in demos[demo_low: demo_high + 1]:
                ep_grp = grp.create_group(f"{demo.split('.')[0]}")

                with h5py.File(os.path.join(args.data_path, demo), 'r') as f:
                    print(f"Adding {demo}")
                    total_num_demos += 1 

                    # preprocess - normalize actions and low dim actions. Resize images
                    d = preprocess(f, stats)
                    
                    for k in d.keys():
                        if k == 'obs':
                            obs_grp = ep_grp.create_group('obs')

                            for obs_k in d[k].keys():
                                if 'depth' in obs_k:
                                    im = d[k][obs_k][()]
                                    if len(im.shape[1:]) == 2:
                                        im = im[..., None]
                                    obs_grp.create_dataset(obs_k, data=im)
                                else:
                                    obs_grp.create_dataset(obs_k, data=d[k][obs_k][()])
                        else:
                            ep_grp.create_dataset(k, data=d[k][()])

                    ep_grp.attrs["num_samples"] = len(d['actions'][()])
                    total_num_samples += len(d['actions'][()])
                    
                if args.render:
                    for i, img in enumerate(d['obs']['agentview_left_image']):
                        cv2.imshow('img', img/255)
                        cv2.imshow('depth', d['obs']['agentview_left_depth'][i])
                        cv2.waitKey(1)

            grp.attrs["num_demos"] = total_num_demos
            grp.attrs["total"] = total_num_samples
            grp.attrs['env_args'] = json.dumps(dict(env_name='tiago_moma', env_kwargs={}, type=1), indent=4)