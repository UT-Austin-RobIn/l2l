from stable_baselines3.common.callbacks import BaseCallback

class StoreRolloutBufferCallback(BaseCallback):
    
    def __init__(self, dataset, verbose: int = 0):
        super().__init__(verbose)
        self.dataset = dataset

    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self):
        self.dataset.add_rollout_buffer(self.model.rollout_buffer)

class StoreReplayBufferCallback(BaseCallback):
    
    def __init__(self, dataset, verbose: int = 0):
        super().__init__(verbose)
        self.dataset = dataset

    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self):
        self.dataset.add_replay_buffer(self.model.replay_buffer)