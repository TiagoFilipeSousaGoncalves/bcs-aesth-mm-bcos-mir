# B-cos DenseNet models
# Modified from https://github.com/pytorch/vision/blob/0504df5ddf9431909130e7788faf05446bb8/torchvision/models/densenet.py



# Imports
import math
import re
from collections import OrderedDict
from typing import Any, Callable, List, Optional, Tuple, Union, Dict
from PIL import Image

# PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor
from torch.hub import load_state_dict_from_url
import torchvision
from torchvision import transforms

# Bcos Imports
import bcos.data.transforms as custom_transforms
from bcos.common import BcosUtilMixin
from bcos.modules import BcosConv2d, LogitLayer, norms
from bcos.modules.common import DetachableModule



# Constants
DEFAULT_NORM_LAYER = norms.NoBias(norms.DetachablePositionNorm2d)
DEFAULT_CONV_LAYER = BcosConv2d
BASE = "https://github.com/B-cos/B-cos-v2/releases/download/v0.0.1-weights"

# Dictionary: Map the BASE into a model name (URL) 
URLS = {
    "bcosdensenet121": f"{BASE}/densenet_121-b8daf96afb.pth",
    "bcosdensenet161": f"{BASE}/densenet_161-9e9ea51353.pth",
    "bcosdensenet169": f"{BASE}/densenet_169-7037ee0604.pth",
    "bcosdensenet201": f"{BASE}/densenet_201-00ac87066f.pth",
    "bcosdensenet121_long": f"{BASE}/densenet_121_long-5175461597.pth"
}



# Class: Constructor of a DenseLayer
class _DenseLayer(nn.Module):
    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        bn_size: int,
        drop_rate: float,
        memory_efficient: bool = False,
        norm_layer: Callable[..., nn.Module] = DEFAULT_NORM_LAYER,
        conv_layer: Callable[..., nn.Module] = DEFAULT_CONV_LAYER,
    ) -> None:
        super(_DenseLayer, self).__init__()
        # Diff to torchvision: Removed ReLU and replaced BatchNorm with norm_layer
        self.norm1 = norm_layer(num_input_features)
        # Diff to torchvision: Replace Conv2d with ProjectionConv2d
        self.conv1 = conv_layer(
            num_input_features,
            bn_size * growth_rate,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.norm2 = norm_layer(bn_size * growth_rate)
        self.conv2 = conv_layer(
            bn_size * growth_rate,
            growth_rate,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        # Diff End
        # Diff End
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        # Diff to torchvision: Deleted relu
        bottleneck_output = self.conv1(self.norm1(concated_features))  # noqa: T484
        # Diff End
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input: List[Tensor]) -> bool:
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input: List[Tensor]) -> Tensor:
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: List[Tensor]) -> Tensor:
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)
        # Diff to torchvision: Deleted relu
        new_features = self.conv2(self.norm2(bottleneck_output))
        # Diff End

        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training
            )
        return new_features



# Class: Constructor of the DenseBlock
class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        memory_efficient: bool = False,
        norm_layer: Callable[..., nn.Module] = DEFAULT_NORM_LAYER,
        conv_layer: Callable[..., nn.Module] = DEFAULT_CONV_LAYER,
    ) -> None:
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                norm_layer=norm_layer,
                conv_layer=conv_layer,
            )
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)



# Class: Construction of the Transition layer
class _Transition(nn.Sequential):
    def __init__(
        self,
        num_input_features: int,
        num_output_features: int,
        norm_layer: Callable[..., nn.Module] = DEFAULT_NORM_LAYER,
        conv_layer: Callable[..., nn.Module] = DEFAULT_CONV_LAYER,
    ) -> None:
        super(_Transition, self).__init__()
        # Diff to torchvision: Deleted relu and changed conv to ProjectionConv2d
        self.add_module("norm", norm_layer(num_input_features))
        self.conv = conv_layer(
            num_input_features,
            num_output_features,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        # Diff End

        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))



# Class: BcosDenseNet
class BcosDenseNet(BcosUtilMixin, nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """

    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0,
        num_classes: int = 1000,
        in_chans: int = 6,
        memory_efficient: bool = False,
        norm_layer: Callable[..., nn.Module] = DEFAULT_NORM_LAYER,
        conv_layer: Callable[..., nn.Module] = DEFAULT_CONV_LAYER,
        small_inputs: bool = False,  # True for 32x32 images from gpleiss' impl
        logit_bias: Optional[float] = None,
        logit_temperature: Optional[float] = None,
    ) -> None:
        super(BcosDenseNet, self).__init__()

        # First convolution
        # Diff to torchvision: Deleted ReLU, changed Conv2d for ProjectionConv2d and
        # MaxPool for AvgPool
        if small_inputs:
            self.features = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv0",
                            conv_layer(
                                in_chans,
                                num_init_features,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                            ),
                        )
                    ]
                )
            )
        else:
            self.features = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv0",
                            conv_layer(
                                in_chans,
                                num_init_features,
                                kernel_size=7,
                                stride=2,
                                padding=3,
                            ),
                        ),
                        ("norm0", norm_layer(num_init_features)),
                        ("pool0", nn.AvgPool2d(kernel_size=3, stride=2, padding=1)),
                    ]
                )
            )
        # Diff End
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                norm_layer=norm_layer,
                conv_layer=conv_layer,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2,
                    norm_layer=norm_layer,
                    conv_layer=conv_layer,
                )
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        # Final norm layer
        self.features.add_module("norm5", norm_layer(num_features))
        self.num_features_ = num_features
        self.logit_temperature_ = logit_temperature
        self.logit_bias_ = logit_bias
        
        self.num_classes = num_classes
        if num_classes == 0:
            self.classifier = nn.Identity()
            self.logit_layer = nn.Identity()
        else:
        # Diff to torchvision: changed Linear layer to BcosConv (conv classifier)
        # self.num_classes = num_classes
            self.classifier = conv_layer(num_features, num_classes, kernel_size=1)
            self.logit_layer = LogitLayer(
                logit_temperature=logit_temperature,
                logit_bias=logit_bias or -math.log(num_classes - 1),
            )
        # Diff End

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        # Diff to torchvision: Deleted relu
        features = self.features(x)
        # out = F.relu(features, inplace=True)
        out = self.classifier(features)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.logit_layer(out)
        # Diff End
        return out

    def get_classifier(self) -> nn.Module:
        """Returns the classifier part of the model. Note this comes before global pooling."""
        return self.classifier

    def get_feature_extractor(self) -> nn.Module:
        """Returns the feature extractor part of the model. Without global pooling."""
        return self.features
    
    # Method: get_transforms
    def get_transform(self):
        def transform(image_path):
            transforms_ = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    custom_transforms.AddInverse(),
                ]
            )
            image = Image.open(image_path).convert('RGB')
            image_trans = transforms_(image)
            return image_trans
        return transform



# TODO: Erase after testing
# Function: Load state dictionary
# def _load_state_dict(model: nn.Module, model_url: str, progress: bool) -> None:
#     # '.'s are no longer allowed in module names, but previous _DenseLayer
#     # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
#     # They are also in the checkpoints in model_urls. This pattern is used
#     # to find such keys.
#     pattern = re.compile(
#         r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
#     )

#     state_dict = load_state_dict_from_url(
#         model_url, map_location="cpu", progress=progress, check_hash=True
#     )
#     for key in list(state_dict.keys()):
#         res = pattern.match(key)
#         if res:
#             new_key = res.group(1) + res.group(2)
#             state_dict[new_key] = state_dict[key]
#             del state_dict[key]
#     model.load_state_dict(state_dict)



# Function: Constructor of a general DenseNet model
def _densenet(
    arch: str,
    growth_rate: int,
    block_config: Union[Tuple[int, int, int], Tuple[int, int, int, int]],
    num_init_features: int,
    pretrained: bool = False,
    weights: str = '',
    progress: bool = False,
    **kwargs: Any,
) -> BcosDenseNet:

    assert arch in ("densenet121", "densenet161", "densenet169", "densenet201"), f"Please provide a valid architecture. {arch} is not valid."

    if pretrained:
        if kwargs['num_classes'] == 1000:
            model = BcosDenseNet(growth_rate, block_config, num_init_features, **kwargs)
            # print(model.state_dict)
            
            assert weights in URLS.keys(), f"Please provide a valid weights string. {weights} is not valid."
            url = URLS[weights]
            
            pattern = re.compile(r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$")
            state_dict = load_state_dict_from_url(url, progress=progress, check_hash=True)
            for key in list(state_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]
            
            # Note: We have to load state_dict with strict=False because of the BatchNorm keys
            model.load_state_dict(state_dict, strict=False)

        else:
            num_classes_ = kwargs['num_classes']
            kwargs['num_classes'] = 1000
            model = BcosDenseNet(growth_rate, block_config, num_init_features, **kwargs)
            # print(model.state_dict().keys())
            
            assert weights in URLS.keys(), f"Please provide a valid weights string. {weights} is not valid."
            url = URLS[weights]
            
            pattern = re.compile(r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$")
            state_dict = load_state_dict_from_url(url, progress=progress, check_hash=True)
            for key in list(state_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]
            
            # Note: We have to load state_dict with strict=False because of the BatchNorm keys
            model.load_state_dict(state_dict, strict=False)
            if num_classes_ == 0:
                model.classifier = nn.Identity()
                model.logit_layer = nn.Identity()
            elif num_classes_ > 0 and num_classes_ != 1000:
                model.classifier = BcosConv2d(model.num_features_, num_classes_, kernel_size=1)
                model.logit_layer = LogitLayer(
                    logit_temperature=model.logit_temperature_,
                    logit_bias=model.logit_bias_ or -math.log(num_classes_ - 1),
                )
    else:
        model = BcosDenseNet(growth_rate, block_config, num_init_features, **kwargs)

    # TODO: Erase after reviewing
    # If we want a pretrained Bcos model on ImageNet
    # if pretrained:
    #     assert weights in URLS.keys(), f"Please provide a valid weights string. {weights} is not valid."
    #     url = URLS[weights]
    #     state_dict = load_state_dict_from_url(url, progress=progress, check_hash=True)
    #     model.load_state_dict(state_dict)

    return model



# Function: bcosdensenet121
def bcosdensenet121(pretrained: bool = False, weights: str = '', progress: bool = False, num_init_features=64, growth_rate=32, **kwargs: Any) -> BcosDenseNet:
    return _densenet(
        arch="densenet121",
        growth_rate=growth_rate, 
        block_config=(6, 12, 24, 16), 
        num_init_features=num_init_features, 
        pretrained=pretrained, 
        weights=weights, 
        progress=progress, 
        **kwargs
        )



# Function: bcosdensenet121_pretr
def bcosdensenet121_pretr(pretrained: bool = True, weights: str = "bcosdensenet121", progress: bool = True, num_init_features=64, growth_rate=32, **kwargs: Any) -> BcosDenseNet:
    return _densenet("densenet121", growth_rate, (6, 12, 24, 16), num_init_features, pretrained, weights, progress, **kwargs)



# Function: bcosdensenet121_pretr_l
def bcosdensenet121_pretr_l(pretrained: bool = True, weights: str = "bcosdensenet121_long", progress: bool = True, num_init_features=64, growth_rate=32, **kwargs: Any) -> BcosDenseNet:
    return _densenet("densenet121", growth_rate, (6, 12, 24, 16), num_init_features, pretrained, weights, progress, **kwargs)



# Function: bcosdensenet161
def bcosdensenet161(pretrained: bool = False, weights: str = '', progress: bool = False, **kwargs: Any) -> BcosDenseNet:
    return _densenet("densenet161", 48, (6, 12, 36, 24), 96, pretrained, weights, progress, **kwargs)



# Function: bcosdensenet161_pretr
def bcosdensenet161_pretr(pretrained: bool = True, weights: str = 'bcosdensenet161', progress: bool = True, **kwargs: Any) -> BcosDenseNet:
    return _densenet("densenet161", 48, (6, 12, 36, 24), 96, pretrained, weights, progress, **kwargs)


# Function: bcosdensenet169
def bcosdensenet169(pretrained: bool = False, weights: str = '', progress: bool = False, **kwargs: Any) -> BcosDenseNet:
    return _densenet("densenet169", 32, (6, 12, 32, 32), 64, pretrained, weights, progress, **kwargs)



# Function: bcosdensenet169_pretr
def bcosdensenet169_pretr(pretrained: bool = True, weights: str = 'bcosdensenet169', progress: bool = True, **kwargs: Any) -> BcosDenseNet:
    return _densenet("densenet169", 32, (6, 12, 32, 32), 64, pretrained, weights, progress, **kwargs)



# Function: bcosdensenet201
def bcosdensenet201(pretrained: bool = False, weights: str = '', progress: bool = False, **kwargs: Any) -> BcosDenseNet:
    return _densenet("densenet201", 32, (6, 12, 48, 32), 64, pretrained, weights, progress, **kwargs)



# Function: bcosdensenet201_pretr
def bcosdensenet201_pretr(pretrained: bool = True, weights: str = 'bcosdensenet201', progress: bool = True, **kwargs: Any) -> BcosDenseNet:
    return _densenet("densenet201", 32, (6, 12, 48, 32), 64, pretrained, weights, progress, **kwargs)



# Class: BaselineDenseNet
class BaselineDenseNet(BcosUtilMixin, nn.Module):
    def __init__(self, depth, num_classes, pretrained=True):
        super(BaselineDenseNet, self).__init__()
        
        # Check if we have a valid DenseNet request
        assert depth in ("densenet121", "densenet161", "densenet169", "densenet201"), "Please provide a valid depth for DenseNet (i.e., 'densenet121', 'densenet161', 'densenet169, 'densenet201')."


        # Load feature extractor
        if depth == "densenet121":
            features = torchvision.models.densenet121(pretrained=pretrained).features
        elif depth == "densenet161":
            features = torchvision.models.densenet161(pretrained=pretrained).features
        elif depth == "densenet169":
            features = torchvision.models.densenet169(pretrained=pretrained).features
        else:
            features = torchvision.models.densenet201(pretrained=pretrained).features


        # Create classification layer
        input_ = torch.rand((1, 3, 224, 224))
        input_ = features(input_)
        input_ = F.adaptive_avg_pool2d(input_, (1, 1))
        in_features = input_.size(0) * input_.size(1) * input_.size(2) * input_.size(3)


        # Assign variables
        self.features = features
        if num_classes == 0:
            self.classifier = nn.Identity()
        else:
            self.classifier = nn.Linear(in_features=in_features, out_features=num_classes)

        return


    # Method: Forward
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=False)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        
        return out
    
    def get_classifier(self) -> nn.Module:
        """Returns the classifier part of the model. Note this comes before global pooling."""
        return self.classifier

    def get_feature_extractor(self) -> nn.Module:
        """Returns the feature extractor part of the model. Without global pooling."""
        return self.features
    
    # Method: get_transforms
    def get_transform(self):
        def transform(image_path):
            transforms_ = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
            image = Image.open(image_path).convert('RGB')
            image_trans = transforms_(image)
            return image_trans
        return transform



# Function: baseline_densenet121
def baseline_densenet121(num_classes, pretrained):
    return BaselineDenseNet(depth="densenet121", num_classes=num_classes, pretrained=pretrained)



# Function: baseline_densenet161
def baseline_densenet161(num_classes, pretrained):
    return BaselineDenseNet(depth="densenet161", num_classes=num_classes, pretrained=pretrained)



# Function: baseline_densenet169
def baseline_densenet169(num_classes, pretrained):
    return BaselineDenseNet(depth="densenet169", num_classes=num_classes, pretrained=pretrained)



# Function: baseline_densenet201
def baseline_densenet201(num_classes, pretrained):
    return BaselineDenseNet(depth="densenet201", num_classes=num_classes, pretrained=pretrained)
