import torch

class BufferDataset(torch.utils.data.Dataset):

    def __init__(self, device, max_len=10000):
        self.rollout_buffer_agg = []
        self.device = device
        self.max_len = max_len

    def __len__(self):
        return len(self.rollout_buffer_agg)
    
    def __getitem__(self, idx):
        obs = self.rollout_buffer_agg[idx]

        keys_to_delete = []
        for k in obs.keys():
            if ('image' not in k) and k!='privileged_info':
                keys_to_delete.append(k)
        for k in keys_to_delete:
            del obs[k]

        return self.rollout_buffer_agg[idx]
    
    def add_rollout_buffer(self, rollout_buffer):
        for batch in rollout_buffer.get(1):
            self.rollout_buffer_agg.append({k:v[0].cpu() for k, v in batch[0].items()})

            if len(self.rollout_buffer_agg) > self.max_len:
                self.rollout_buffer_agg.pop(0)

    def add_replay_buffer(self, replay_buffer):
        self.add_rollout_buffer(replay_buffer)

