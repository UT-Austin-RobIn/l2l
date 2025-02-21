import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models as vision_models
from l2l.imitation.utils.torch_utils import Interpolate

# Resnet
class ResNet18(nn.Module):
    """
    A ResNet18 block that can be used to process input images.
    """
    def __init__(self, input_channel=3):
        super(ResNet18, self).__init__()
        net = vision_models.resnet18(weights=None)

        if input_channel != 3:
            net.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # cut the last fc layer
        self._input_channel = input_channel
        self.nets = torch.nn.Sequential(*(list(net.children())[:-2]))

    def output_shape(self, input_shape):
        assert(len(input_shape) == 3)
        out_h = int(math.ceil(input_shape[1] / 32.))
        out_w = int(math.ceil(input_shape[2] / 32.))
        return [512, out_h, out_w]

    def forward(self, inputs):
        out = self.nets(inputs)
        return out

# Resnet
class ResNet34(nn.Module):
    """
    A ResNet18 block that can be used to process input images.
    """
    def __init__(self, input_channel=3):
        super(ResNet34, self).__init__()
        net = vision_models.resnet34(weights=None)

        if input_channel != 3:
            net.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # cut the last fc layer
        self._input_channel = input_channel
        self.nets = torch.nn.Sequential(*(list(net.children())[:-2]))

    def output_shape(self, input_shape):
        assert(len(input_shape) == 3)
        out_h = int(math.ceil(input_shape[1] / 32.))
        out_w = int(math.ceil(input_shape[2] / 32.))
        return [512, out_h, out_w]

    def forward(self, inputs):
        out = self.nets(inputs)
        return out
    
class R3M(nn.Module):

    def __init__(self, input_channel=3, r3m_model_class='resnet18', freeze=True):
        super(R3M, self).__init__()
        
        try:
            from r3m import load_r3m
        except ImportError:
            raise ImportError('Please install r3m package')

        assert input_channel == 3, 'r3m only supports 3 channel images'
        assert r3m_model_class in ['resnet18', 'resnet34', 'resnet50'], 'r3m only supports resnet18 and resnet34'
        self.r3m = load_r3m(r3m_model_class)

        self.preprocess = nn.Sequential(
            transforms.Resize((224, 224)),
        )

        self.freeze = freeze
        self.r3m_model_class = r3m_model_class

    def train(self, mode):
        if self.freeze:
            self.r3m.eval()
        else:
            self.r3m.train(mode)

    def output_shape(self, input_shape):
        if self.r3m_model_class == 'resnet50':
            out_dim = 2048
        else:
            out_dim = 512

        return [out_dim, 1, 1] 

    def forward(self, inputs):
        out = self.preprocess(inputs)*255
        return self.r3m(out)

# Resnet
class EfficientNet(nn.Module):
    """
    A ResNet18 block that can be used to process input images.
    """
    def __init__(self, input_channel=3):
        super(EfficientNet, self).__init__()
        net = vision_models.efficientnet_b0(weights=None)

        if input_channel != 3:
            net.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # cut the last fc layer
        self._input_channel = input_channel
        self.nets = torch.nn.Sequential(*(list(net.children())[:-2]))

    def output_shape(self, input_shape):
        assert(len(input_shape) == 3)
        out_h = int(math.ceil(input_shape[1] / 32.))
        out_w = int(math.ceil(input_shape[2] / 32.))
        return [1280, out_h, out_w]

    def forward(self, inputs):
        out = self.nets(inputs)
        return out


# ViT
class VisionTransformer(nn.Module):

    def __init__(
            self,
            input_channel=3,
            model_name='vit_base_patch16_clip_224.openai',
            pretrained=False,
            frozen=False
        ):
        
        super(VisionTransformer, self).__init__()
        
        assert input_channel == 3, 'ViT only supports 3 channel images'
        
        import timm
        model = timm.create_model(
                model_name=model_name,
                pretrained=pretrained,
                global_pool='', # '' means no pooling
                num_classes=0            # remove classification layer
            )
        # model = vision_models.vit_b_16()
        
        if frozen:
            assert pretrained, 'model must be pretrained to be frozen'
            for param in model.parameters():
                param.requires_grad = False
        
        self.nets = nn.Sequential(
            transforms.Resize((224, 224)),
            model
        )

    def output_shape(self, input_shape):
        return [768]

    def forward(self, inputs):
        out = self.nets(inputs)[:, 0, :] # only use th CLS token
        return out

class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax Layer.

    Based on Deep Spatial Autoencoders for Visuomotor Learning by Finn et al.
    https://rll.berkeley.edu/dsae/dsae.pdf
    """
    def __init__(
        self,
        input_shape,
        num_kp=32,
        temperature=1.,
        learnable_temperature=False,
        output_variance=False,
        noise_std=0.0,
    ):
        """
        Args:
            input_shape (list): shape of the input feature (C, H, W)
            num_kp (int): number of keypoints (None for not using spatialsoftmax)
            temperature (float): temperature term for the softmax.
            learnable_temperature (bool): whether to learn the temperature
            output_variance (bool): treat attention as a distribution, and compute second-order statistics to return
            noise_std (float): add random spatial noise to the predicted keypoints
        """
        super(SpatialSoftmax, self).__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape # (C, H, W)

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._num_kp = num_kp
        else:
            self.nets = None
            self._num_kp = self._in_c
        self.learnable_temperature = learnable_temperature
        self.output_variance = output_variance
        self.noise_std = noise_std

        if self.learnable_temperature:
            # temperature will be learned
            temperature = torch.nn.Parameter(torch.ones(1) * temperature, requires_grad=True)
            self.register_parameter('temperature', temperature)
        else:
            # temperature held constant after initialization
            temperature = torch.nn.Parameter(torch.ones(1) * temperature, requires_grad=False)
            self.register_buffer('temperature', temperature)

        pos_x, pos_y = np.meshgrid(
                np.linspace(-1., 1., self._in_w),
                np.linspace(-1., 1., self._in_h)
                )
        pos_x = torch.from_numpy(pos_x.reshape(1, self._in_h * self._in_w)).float()
        pos_y = torch.from_numpy(pos_y.reshape(1, self._in_h * self._in_w)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

        self.kps = None

    def __repr__(self):
        """Pretty print network."""
        header = format(str(self.__class__.__name__))
        return header + '(num_kp={}, temperature={}, noise={})'.format(
            self._num_kp, self.temperature.item(), self.noise_std)

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert(len(input_shape) == 3)
        assert(input_shape[0] == self._in_c)
        return [self._num_kp, 2]

    def forward(self, feature):
        """
        Forward pass through spatial softmax layer. For each keypoint, a 2D spatial 
        probability distribution is created using a softmax, where the support is the 
        pixel locations. This distribution is used to compute the expected value of 
        the pixel location, which becomes a keypoint of dimension 2. K such keypoints
        are created.

        Returns:
            out (torch.Tensor or tuple): mean keypoints of shape [B, K, 2], and possibly
                keypoint variance of shape [B, K, 2, 2] corresponding to the covariance
                under the 2D spatial softmax distribution
        """
        assert(feature.shape[1] == self._in_c)
        assert(feature.shape[2] == self._in_h)
        assert(feature.shape[3] == self._in_w)
        if self.nets is not None:
            feature = self.nets(feature)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        feature = feature.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(feature / self.temperature, dim=-1)
        # [1, H * W] x [B * K, H * W] -> [B * K, 1] for spatial coordinate mean in x and y dimensions
        expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)
        # stack to [B * K, 2]
        expected_xy = torch.cat([expected_x, expected_y], 1)
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._num_kp, 2)

        if self.training:
            noise = torch.randn_like(feature_keypoints) * self.noise_std
            feature_keypoints += noise

        if self.output_variance:
            # treat attention as a distribution, and compute second-order statistics to return
            expected_xx = torch.sum(self.pos_x * self.pos_x * attention, dim=1, keepdim=True)
            expected_yy = torch.sum(self.pos_y * self.pos_y * attention, dim=1, keepdim=True)
            expected_xy = torch.sum(self.pos_x * self.pos_y * attention, dim=1, keepdim=True)
            var_x = expected_xx - expected_x * expected_x
            var_y = expected_yy - expected_y * expected_y
            var_xy = expected_xy - expected_x * expected_y
            # stack to [B * K, 4] and then reshape to [B, K, 2, 2] where last 2 dims are covariance matrix
            feature_covar = torch.cat([var_x, var_xy, var_xy, var_y], 1).reshape(-1, self._num_kp, 2, 2)
            feature_keypoints = (feature_keypoints, feature_covar)

        if isinstance(feature_keypoints, tuple):
            self.kps = (feature_keypoints[0].detach(), feature_keypoints[1].detach())
        else:
            self.kps = feature_keypoints.detach()
        return feature_keypoints

class ImageDecoder(nn.Module):

    def __init__(self, input_dim, img_shape):
        super().__init__()

        start_channels = 4*128
        net = [nn.Linear(input_dim, 4*2048), nn.ReLU(), nn.Unflatten(1, (start_channels, 4, 4))]
        
        c, w, h = img_shape
        assert w == h, 'width and height must be the same'
        
        size_upper_bound = (int(np.log2(w)) - 2) + 1

        for i in range(size_upper_bound):
            out_channels = max(4, start_channels//2)

            net.append(Interpolate(scale_factor=2)) 
            net.append(nn.Conv2d(in_channels=start_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
            net.append(nn.BatchNorm2d(out_channels))
            net.append(nn.LeakyReLU(0.2))

            start_channels = max(4, start_channels//2)
        
        net.extend([
                    Interpolate(size=(w, h)),
                    nn.Conv2d(in_channels=start_channels, out_channels=c, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(in_channels=c, out_channels=10, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(in_channels=10, out_channels=c, kernel_size=3, stride=1, padding=1),
                    nn.Sigmoid()
                ])

        self.net = nn.Sequential(*net)

    def forward(self, x):
        sh = x.shape
        if len(sh) == 3:
            x = x.view(-1, sh[-1])
        out = self.net(x)
        out = out.view(*sh[:2], *out.shape[1:])
        return out



class AutoEncoders(nn.Module):
    def __init__(self, img_shape, encoder):
        super().__init__()

        self.encoder = encoder
        encoder_shape = self.encoder.output_shape(img_shape)

        self.decoder = ImageDecoder(np.prod(encoder_shape), img_shape)

    def forward(self, x):
        x = self.encoder(x)
        # x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        return self.decoder(x)

if __name__=="__main__":

    images = torch.rand(8, 10, 3, 84, 84)
    model = VisionTransformer()
    # model = VisionTransformer('vit_base_patch16_clip_224.openai', pretrained=False, frozen=False)
    x = model(images)
    print(x.shape)
    
    # pool = SpatialSoftmax([512, 3, 3], num_kp=32)
    # print(pool(x).shape)

    