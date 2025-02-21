import cv2
import os
import numpy as np
import torch
from torch import nn
import torch.distributions as D

from l2l.utils.general_utils import AttrDict

from l2l.imitation.models.image_nets import ResNet18
from l2l.imitation.models.obs_nets import VisionCore
from l2l.imitation.algo.base_algo import BaseAlgo
from l2l.imitation.utils.tensor_utils import recursive_dict_list_tuple_apply
from l2l.imitation.algo.bc_ce import BC_CrossEntropy
from l2l.imitation.utils.torch_utils import freeze_module
from l2l.imitation.models.distribution_nets import MDN, Gaussian
from l2l.imitation.models.obs_nets import VisionCore
from l2l.imitation.models.image_nets import ResNet18
from l2l.wrappers.policy_wrappers import UncertaintyPolicyWrapper, UncertaintyVisuomotorPolicyWrapper

DEBUG=False

class Image2ActionModel(BaseAlgo):

    def __init__(self, config, loading_ckpt=False):
        super(Image2ActionModel, self).__init__()
        self.config = config

        self.image_key = config.image_key
        self.privileged_key = config.privileged_key

        self.image_shape = config.image_shape

        self.discrete_privileged = config.get('discrete_privileged', True)

        self.privileged_classes = config.privileged_classes
        self.privileged_dim = config.get('privileged_dim', -1)
        self.gmm_pi = config.get('gmm_head_for_pi', False)

        self.n_ensemble = config.n_ensemble

        self.setup_model(loading_ckpt=loading_ckpt)
    
    def setup_model(self, loading_ckpt=False):
        self.nets = nn.ModuleDict()

        if self.discrete_privileged:
            output_layer = nn.Linear(64, self.privileged_classes)  # maybe replace with MLP
        else:
            if self.gmm_pi:
                output_layer = MDN(64, self.privileged_dim, 10, has_time_dimension=True) # pi classes captures the number of possible pi
            else:
                output_layer = nn.Linear(64, self.privileged_dim)

        
        self.nets["obs_encoder"] = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(3136, 64),
                    nn.ReLU(),
                    output_layer,
            ) for _ in range(self.n_ensemble)
        ])
        
        # do special loading when loading a ckpt so the ckpt path doesn't matter
        if not loading_ckpt:
            kwargs, state = torch.load(self.config.bc_ckpt)
            self.config.bc_config = kwargs['config']
            self.policy = BC_CrossEntropy(self.config.bc_config)
            self.policy.load_state_dict(state)
        else:
            self.policy = BC_CrossEntropy(self.config.bc_config)
    
        freeze_module(self.policy)
        self.policy.eval()

    def get_optimizers_and_schedulers(self, **kwargs):
        optimizer = torch.optim.Adam(self.nets.parameters(), lr=3e-4, weight_decay=1e-7)
        lr_scheduler = None
    
        return [optimizer], [lr_scheduler]

    def forward(self, batch):
        image_batch = batch[self.image_key]
        assert len(image_batch.shape) == 4, f"Image missing a dimension. Recieved shape: {image_batch.shape}"
        assert image_batch.shape[1] == 3, f"Image should have 3 channels. Recieved shape: {image_batch.shape}"

        if DEBUG and self.training:
            img_to_display = image_batch[0].permute(1, 2, 0).detach().cpu().numpy()
            cv2.imshow("Image", img_to_display)
            cv2.waitKey(1)
        
        embedding = []
        idx = np.random.randint(0, self.n_ensemble)
        for i, net in enumerate(self.nets["obs_encoder"]):
            if self.training:
                if i == idx or (np.random.rand() < 0.5):
                    embedding.append(net(image_batch))
            else:
                embedding.append(net(image_batch))

        if not self.discrete_privileged and self.gmm_pi:
            return embedding
        return torch.cat(embedding, dim=0)

    def process_pi(self, privileged_batch):
        # TODO: right now all pi processing is done in the environment
        return privileged_batch

    def compute_loss(self, batch):
        privileged_batch = batch[self.privileged_key].clone()
        privileged_batch = self.process_pi(privileged_batch)

        batch = self.preprocess_batch(batch)
        embedding = self.forward(batch)

        
        if DEBUG and self.training:
            if 'camera_angle' in batch.keys():
                print("Total Blocks:", 
                    torch.sum(torch.logical_and(batch['camera_angle'] < 0.6, batch['camera_angle'] > 0.3)).item(),
                    torch.sum(torch.logical_and(batch['camera_angle'] > -0.6, batch['camera_angle'] < -0.3)).item())
            if 'stage_id' in batch.keys():
                if batch['stage_id'].shape[1] == 3:
                    print("Stage ID:", torch.sum(batch['stage_id'][:, 0]==1).item(), torch.sum(batch['stage_id'][:, 1]==1).item(), torch.sum(batch['stage_id'][:, 2]==1).item())
                elif batch['stage_id'].shape[1] == 2:
                    print("Stage ID:", torch.sum(batch['stage_id'][:, 0]==1).item(), torch.sum(batch['stage_id'][:, 1]==1).item())

        losses = AttrDict(total = 0)

        if self.discrete_privileged:
            losses.embedding_loss = nn.CrossEntropyLoss()(embedding, privileged_batch.repeat(embedding.shape[0]//privileged_batch.shape[0], 1))
        else:
            if self.gmm_pi:
                log_probs = [emb.log_prob(privileged_batch) for emb in embedding]
                log_probs = torch.stack(log_probs, dim=0)
                losses.embedding_loss = -log_probs.mean()

            else:
                losses.embedding_loss = nn.MSELoss()(embedding, privileged_batch.repeat(embedding.shape[0]//privileged_batch.shape[0], 1))
        
        losses.total += losses.embedding_loss
        return losses 
    
    def f(self, batch):
        batch = self.preprocess_batch(batch)
        return self.forward(batch)

    def clone_batch(self, batch):
        _batch = {}
        for k in batch.keys():
            _batch[k] = batch[k].clone()
        return _batch

    def preprocess_batch(self, batch):
        _batch = {}
        for k in batch.keys():
            if 'image' in k:
                _batch[k] = batch[k].permute(0, 3, 1, 2)/255
            else:
                _batch[k] = batch[k]
        return _batch

    def preprocess_obs(self, obs):
        _obs = {}
        for k in obs.keys():
            if isinstance(obs[k], torch.Tensor) or isinstance(obs[k], np.ndarray):
                _obs[k] = obs[k]
            else:
                _obs[k] = np.array([obs[k]])
        
        _obs = recursive_dict_list_tuple_apply(
                _obs,
                {
                    torch.Tensor: lambda x: x[None].clone().float().to(self.device),
                    np.ndarray: lambda x: torch.from_numpy(x.astype(np.float32).copy())[None].float().to(self.device),
                    type(None): lambda x: x,
                }
            )
        return _obs

    def sample_privileged(self, embeddings):
        if self.discrete_privileged:
            privileged = torch.distributions.Categorical(logits=embeddings).sample()
            privileged = nn.functional.one_hot(privileged, num_classes=self.privileged_classes)
        else:
            if self.gmm_pi:
                privileged = [embedding.sample() for embedding in embeddings]# since embeddings are already distributions
                privileged = torch.vstack(privileged)
            else:
                privileged = embeddings
        return privileged

    @torch.no_grad()
    def get_reward(self, obs):
        self.eval()
        policy = UncertaintyPolicyWrapper(self.policy)
        batch = self.preprocess_batch(self.preprocess_obs(obs))
        
        gt_actions = torch.argmax(policy.forward({'obs': batch})['action_logits'].detach().mean(dim=0), dim=-1)

        batch = self.preprocess_batch(self.preprocess_obs(obs))                   
        embeddings = self.forward(batch)

        pi_length = self.privileged_classes if self.discrete_privileged else self.privileged_dim
        privileged_batch = torch.cat([self.sample_privileged(embeddings) for _ in range(10)]).reshape(-1, 1, pi_length)
 
        action_loss = policy.get_loss_from_sampled_privileged(batch, privileged_batch, gt_actions)
        self.train()

        return action_loss#, gt_actions
    
    def get_outputs(self, obs):
        outputs = self.get_action_and_uncertainty(obs)

        return outputs

    def compute_uncertainty(self, actions):
        def categorical_kl_divergence(p, q):
            return np.sum(p*np.log(p/q))
        
        act_prob = np.exp(actions)/np.sum(np.exp(actions), axis=-1).reshape(-1, 1)
        
        all_kl = []
        for i, p in enumerate(act_prob):
            for j, q in enumerate(act_prob):
                if i == j:
                    continue
                kl = categorical_kl_divergence(p, q)
                all_kl.append(kl)
        return np.mean(all_kl)

    @torch.no_grad()
    def get_action_and_uncertainty(self, obs):
        
        self.eval()
        policy = UncertaintyPolicyWrapper(self.policy)

        batch = self.preprocess_batch(self.preprocess_obs(obs))               
        embeddings = self.forward(batch)

        pi_length = self.privileged_classes if self.discrete_privileged else self.privileged_dim
        privileged_batch = torch.cat([self.sample_privileged(embeddings) for _ in range(10)]).reshape(-1, 1, pi_length)
        
        actions = policy.get_actions_from_sampled_privileged(batch, privileged_batch).detach().cpu().numpy()
        actions = actions.reshape(-1, actions.shape[-1])

        uncertainty = self.compute_uncertainty(actions) 
        # print(privileged_batch.float().mean(dim=0), actions.mean(axis=0), uncertainty, obs['stage_id'])
    
        # use 0.01 as th for language and 0.5 otherwise 
        is_uncertain = uncertainty > 0.5
        action = np.argmax(actions.mean(axis=0), axis=-1)
        self.train()

        return action, is_uncertain, uncertainty
        
    @staticmethod
    def load_weights(path):
        if os.path.exists(path):
            kwargs, state = torch.load(path)
            config = kwargs['config']
            device = kwargs['device']

            model = Image2ActionModel(config, loading_ckpt=True)
            model.load_state_dict(state)
            return model
        else:
            print("File not found: {}".format(path))
            return False


# exclusively used for real world evals
try:
    from l2l.imitation.models.data_augmentation import TranslationAug
except:
    pass
from l2l.imitation.algo.bc_transformer import BCTransformer
class TransformerImage2ActionModelLarge(Image2ActionModel):

    def __init__(self, config):
        super(TransformerImage2ActionModelLarge, self).__init__(config)

        self.last_privileged_sample = None
        self.last_uncertain = True

    def setup_model(self):
        self.nets = nn.ModuleDict()

        augmentation = TranslationAug(
            input_shape=self.image_shape,
            translation=15
        )

        self.nets["obs_encoder"] = nn.ModuleList([
            nn.Sequential(
                augmentation,
                VisionCore(input_shape=self.image_shape, backbone_class=ResNet18, pool_class=None, feature_dim=64),
                nn.Linear(64, self.privileged_classes), # maybe replace with MLP
            ) for _ in range(self.n_ensemble)
        ])

        self.policy = BCTransformer.load_weights(self.config.bc_ckpt)
        freeze_module(self.policy)
        self.policy.eval()

    def preprocess_batch(self, batch):
        _batch = {}
        for k in batch.keys():
            if 'image' in k:
                if len(batch[k].shape) == 4:
                    _batch[k] = batch[k].clone().permute(0, 3, 1, 2)/255
                else:
                    _batch[k] = batch[k].clone().permute(0, 1, 4, 2, 3)/255
            else:
                _batch[k] = batch[k].clone()
        return _batch

    def preprocess_obs(self, obs):
        _obs = {}
        for k in obs.keys():
            if isinstance(obs[k], torch.Tensor) or isinstance(obs[k], np.ndarray):
                _obs[k] = obs[k]
            else:
                _obs[k] = np.array([obs[k]])
        
        _obs = recursive_dict_list_tuple_apply(
                _obs,
                {
                    torch.Tensor: lambda x: x[None, None].float().to(self.device),
                    np.ndarray: lambda x: torch.from_numpy(x.astype(np.float32).copy())[None, None].float().to(self.device),
                    type(None): lambda x: x,
                }
            )
        return _obs
    
    @torch.no_grad()
    def get_reward(self, obs):
        self.eval()
        policy = UncertaintyVisuomotorPolicyWrapper(self.policy, self.privileged_key)
        batch = self.preprocess_batch(self.preprocess_obs(obs))
        
        gt_actions = policy.forward({'obs': batch})['action_dist']

        batch = self.preprocess_batch(self.preprocess_obs(obs))
        _batch = {}
        for k in batch.keys():
            _batch[k] = batch[k][:, 0]           
        embeddings = self.forward(_batch)
        
        pi_length = self.privileged_classes if self.discrete_privileged else self.privileged_dim
        privileged_batch = torch.cat([self.sample_privileged(embeddings) for _ in range(10)]).reshape(-1, 1, pi_length)

        action_loss = policy.get_loss_from_sampled_privileged(batch, privileged_batch, gt_actions)
        self.train()

        return action_loss
    
    def compute_uncertainty_gmm(self, pi_logits, means, sigma):
        def sampling_based_kl(p, q):
            samples = p.sample((10,))
            return (p.log_prob(samples) - q.log_prob(samples)).mean()

        all_kl = []
        for i, (l1, m1, s1) in enumerate(zip(pi_logits, means, sigma)):
            p = D.MixtureSameFamily(mixture_distribution=D.Categorical(logits=l1),
                                    component_distribution=D.Independent(D.Normal(loc=m1, scale=s1), 1))

            for j, (l2, m2, s2) in enumerate(zip(pi_logits[i:], means[i:], sigma[i:])):
                if i == j:
                    continue
                q = D.MixtureSameFamily(mixture_distribution=D.Categorical(logits=l2),
                                        component_distribution=D.Independent(D.Normal(loc=m2, scale=s2), 1))
                
                kl = sampling_based_kl(p, q)
                all_kl.append(kl)
        
        # print('disagreeing pairs', np.sum(np.array(all_kl) > 1e5), np.log10(np.mean(all_kl)))

        # return np.sum(np.array(all_kl) > 1e10)
        return np.mean(all_kl)

    @torch.no_grad()
    def get_action_and_uncertainty(self, obs):
        
        self.eval()
        policy = UncertaintyVisuomotorPolicyWrapper(self.policy, self.privileged_key)

        batch = self.preprocess_batch(self.preprocess_obs(obs)) 
        _batch = {}
        for k in batch.keys():
            _batch[k] = batch[k][:, 0]               
        embeddings = self.forward(_batch)

        pi_length = self.privileged_classes if self.discrete_privileged else self.privileged_dim
        privileged_batch = torch.cat([self.sample_privileged(embeddings) for _ in range(10)]).reshape(-1, 1, pi_length)

        ### GMM ###
        gmm, (pi_logits, means, sigma) = policy.get_actions_from_sampled_privileged(batch, privileged_batch)
        uncertainty = self.compute_uncertainty_gmm(pi_logits[:, 0], means[:, 0], sigma[:, 0]) 
        action = gmm.sample().mean(dim=0, keepdim=True)
        uncertain = uncertainty > 1e10
        action = policy.policy.nets['normalizer'].denormalize_by_key(action, 'actions')[0, 0].cpu().numpy()
        ############

        ### Temporal Action Sampling ###
        # if self.last_uncertain or self.last_privileged_sample is None:
        #     policy.policy.reset()
            
        #     p = torch.sum(privileged_batch, dim=0, keepdim=True)
        #     idx = torch.argmax(p, dim=-1)
        #     self.last_privileged_sample = torch.zeros_like(p)
        #     self.last_privileged_sample[..., idx] = 1
                
        # batch = self.preprocess_obs(obs)
        # batch[self.privileged_key] = self.last_privileged_sample.clone()
        # action = policy.get_action(batch)

        # self.last_uncertain = uncertain
        ############################

        # print(uncertainty)
        self.train()

        return action, uncertain, uncertainty

    @staticmethod
    def load_weights(path):
        if os.path.exists(path):
            kwargs, state = torch.load(path)
            config = kwargs['config']
            device = kwargs['device']
            
            model = TransformerImage2ActionModelLarge(config)
            model.load_state_dict(state)
            return model
        else:
            print("File not found: {}".format(path))
            return False
        
    @torch.no_grad()
    def get_action_and_uncertainty_for_real(self, obs_act, obs_look):
        
        self.eval()
        policy = UncertaintyVisuomotorPolicyWrapper(self.policy, self.privileged_key)

        batch_act = self.preprocess_batch(self.preprocess_obs(obs_act))
        batch_look = self.preprocess_batch(self.preprocess_obs(obs_look)) 
        _batch_look = {}
        for k in batch_look.keys():
            _batch_look[k] = batch_look[k][:, 0]               
        embeddings = self.forward(_batch_look)

        pi_length = self.privileged_classes if self.discrete_privileged else self.privileged_dim
        privileged_batch = torch.cat([self.sample_privileged(embeddings) for _ in range(10)]).reshape(-1, 1, pi_length)

        ### GMM ###
        gmm, (pi_logits, means, sigma) = policy.get_actions_from_sampled_privileged(batch_act, privileged_batch)
        uncertainty = self.compute_uncertainty_gmm(pi_logits[:, 0], means[:, 0], sigma[:, 0]) 
        is_uncertain = uncertainty > 1e5

        # action = gmm.sample().mean(dim=0, keepdim=True)
        # action = policy.policy.nets['normalizer'].denormalize_by_key(action, 'actions')[0, 0].cpu().numpy()
        ############

        ### Temporal Action Sampling ###
        if self.last_uncertain or self.last_privileged_sample is None:
            policy.policy.reset()
            
            p = torch.sum(privileged_batch, dim=0, keepdim=True)
            idx = torch.argmax(p, dim=-1)
            self.last_privileged_sample = torch.zeros_like(p)
            self.last_privileged_sample[..., idx] = 1
                
        # batch = self.preprocess_obs(obs_act)
        # batch[self.privileged_key] = self.last_privileged_sample.clone()
        # action = policy.get_action(batch)

        # print(self.last_privileged_sample)
        # print("length of queue", len(self.policy.latent_queue))

        self.last_uncertain = is_uncertain
        ############################

        # print(uncertainty)
        self.train()

        return None, is_uncertain, uncertainty
    
    def get_fast_actions(self, obs_act):
        obs_act = obs_act
        print('last_priv', self.last_privileged_sample[0, 0])
        obs_act[self.privileged_key] = self.last_privileged_sample.clone().numpy()[0, 0]
        action = self.policy.get_action(obs_act)

        return action
    

class TransformerImage2ActionModel(Image2ActionModel):

    def __init__(self, config):
        super(TransformerImage2ActionModel, self).__init__(config)

        self.last_privileged_sample = None
        self.last_uncertain = True

    def setup_model(self):
        self.nets = nn.ModuleDict()
        if self.discrete_privileged:
            output_layer = nn.Linear(64, self.privileged_classes)  # maybe replace with MLP
        else:
            if self.gmm_pi:
                output_layer = MDN(64, self.privileged_dim, 10, has_time_dimension=True) # pi classes captures the number of possible pi
            else:
                output_layer = nn.Linear(64, self.privileged_dim)

        
        self.nets["obs_encoder"] = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(3136, 64),
                    nn.ReLU(),
                    output_layer,
            ) for _ in range(self.n_ensemble)
        ])

        self.policy = BCTransformer.load_weights(self.config.bc_ckpt)
        freeze_module(self.policy)
        self.policy.eval()

    def preprocess_batch(self, batch):
        _batch = {}
        for k in batch.keys():
            if 'image' in k:
                if len(batch[k].shape) == 4:
                    _batch[k] = batch[k].clone().permute(0, 3, 1, 2)/255
                else:
                    _batch[k] = batch[k].clone().permute(0, 1, 4, 2, 3)/255
            else:
                _batch[k] = batch[k].clone()
        return _batch

    def preprocess_obs(self, obs):
        _obs = {}
        for k in obs.keys():
            if isinstance(obs[k], torch.Tensor) or isinstance(obs[k], np.ndarray):
                _obs[k] = obs[k]
            else:
                _obs[k] = np.array([obs[k]])
        
        _obs = recursive_dict_list_tuple_apply(
                _obs,
                {
                    torch.Tensor: lambda x: x[None, None].float().to(self.device),
                    np.ndarray: lambda x: torch.from_numpy(x.astype(np.float32).copy())[None, None].float().to(self.device),
                    type(None): lambda x: x,
                }
            )
        return _obs
    
    @torch.no_grad()
    def get_reward(self, obs):
        self.eval()
        policy = UncertaintyVisuomotorPolicyWrapper(self.policy, self.privileged_key)
        batch = self.preprocess_batch(self.preprocess_obs(obs))
        
        gt_actions = policy.forward({'obs': batch})['action_dist']

        batch = self.preprocess_batch(self.preprocess_obs(obs))
        _batch = {}
        for k in batch.keys():
            _batch[k] = batch[k][:, 0]           
        embeddings = self.forward(_batch)
        
        pi_length = self.privileged_classes if self.discrete_privileged else self.privileged_dim
        privileged_batch = torch.cat([self.sample_privileged(embeddings) for _ in range(10)]).reshape(-1, 1, pi_length)
        # privileged_batch = torch.cat([_batch[self.privileged_key]+torch.randn(4, device='cuda')*0.01 for _ in range(10)]).reshape(-1, 1, pi_length)
        # print(privileged_batch.shape)

        action_loss = policy.get_loss_from_sampled_privileged(batch, privileged_batch, gt_actions)
        self.train()

        return action_loss
    
    def compute_uncertainty_gmm(self, pi_logits, means, sigma):
        def sampling_based_kl(p, q):
            samples = p.sample((10,))
            return (p.log_prob(samples) - q.log_prob(samples)).mean()

        all_kl = []
        for i, (l1, m1, s1) in enumerate(zip(pi_logits, means, sigma)):
            p = D.MixtureSameFamily(mixture_distribution=D.Categorical(logits=l1),
                                    component_distribution=D.Independent(D.Normal(loc=m1, scale=s1), 1))

            for j, (l2, m2, s2) in enumerate(zip(pi_logits[i:], means[i:], sigma[i:])):
                if i == j:
                    continue
                q = D.MixtureSameFamily(mixture_distribution=D.Categorical(logits=l2),
                                        component_distribution=D.Independent(D.Normal(loc=m2, scale=s2), 1))
                
                kl = sampling_based_kl(p, q)
                all_kl.append(kl)
        
        # print('disagreeing pairs', np.sum(np.array(all_kl) > 1e5), np.log10(np.mean(all_kl)))

        # return np.sum(np.array(all_kl) > 1e10)
        return np.mean(all_kl)

    @torch.no_grad()
    def get_action_and_uncertainty(self, obs):
        
        self.eval()
        policy = UncertaintyVisuomotorPolicyWrapper(self.policy, self.privileged_key)

        batch = self.preprocess_batch(self.preprocess_obs(obs)) 
        _batch = {}
        for k in batch.keys():
            _batch[k] = batch[k][:, 0]               
        embeddings = self.forward(_batch)

        pi_length = self.privileged_classes if self.discrete_privileged else self.privileged_dim
        privileged_batch = torch.cat([self.sample_privileged(embeddings) for _ in range(10)]).reshape(-1, 1, pi_length)

        ### GMM ###
        gmm, (pi_logits, means, sigma) = policy.get_actions_from_sampled_privileged(batch, privileged_batch)
        uncertainty = self.compute_uncertainty_gmm(pi_logits[:, 0].cpu(), means[:, 0].cpu(), sigma[:, 0].cpu()) 
        action = gmm.sample()[0][None]#.mean(dim=0, keepdim=True)
        uncertain = uncertainty > 1e10
        action = policy.policy.nets['normalizer'].denormalize_by_key(action, 'actions')[0, 0].cpu().numpy()
        ############

        ### Temporal Action Sampling ###
        # if self.last_uncertain or self.last_privileged_sample is None:
        #     policy.policy.reset()
            
        #     p = torch.sum(privileged_batch, dim=0, keepdim=True)
        #     idx = torch.argmax(p, dim=-1)
        #     self.last_privileged_sample = torch.zeros_like(p)
        #     self.last_privileged_sample[..., idx] = 1
                
        # batch = self.preprocess_obs(obs)
        # batch[self.privileged_key] = self.last_privileged_sample.clone()
        # action = policy.get_action(batch)

        # self.last_uncertain = uncertain
        ############################

        # print(uncertainty)
        self.train()

        return action, uncertain, uncertainty

    @staticmethod
    def load_weights(path):
        if os.path.exists(path):
            kwargs, state = torch.load(path)
            config = kwargs['config']
            device = kwargs['device']
            
            model = TransformerImage2ActionModel(config)
            model.load_state_dict(state)
            return model
        else:
            print("File not found: {}".format(path))
            return False
        
    @torch.no_grad()
    def get_gt_action(self, obs):
        self.eval()
        policy = UncertaintyVisuomotorPolicyWrapper(self.policy, self.privileged_key)
        batch = self.preprocess_batch(self.preprocess_obs(obs))
        
        action = policy.forward({'obs': batch})['action_dist'].sample()
        action = policy.policy.nets['normalizer'].denormalize_by_key(action, 'actions')[0, 0].cpu().numpy()

        return action