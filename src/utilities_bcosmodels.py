# Imports
from PIL import Image
import timm

# PyTorch Imports
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights, vgg16, VGG16_Weights

# BCos Imports
from bcos.models.convnext import (
    convnext_atto,
    convnext_tiny,
    convnext_small,
    convnext_base,
    convnext_large
)
from bcos.models.densenet import (
    baseline_densenet121,
    baseline_densenet161,
    baseline_densenet169,
    baseline_densenet201,
    densenet121,
    densenet161,
    densenet169,
    densenet201,
)
from bcos.models.resnet import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    resnext50_32x4d,
    resnext101_32x8d,
    wide_resnet50_2,
    wide_resnet101_2
)
from bcos.models.vgg import (
    vgg11,
    vgg11_bnu,
    vgg13,
    vgg13_bnu,
    vgg16,
    vgg16_bnu,
    vgg19,
    vgg19_bnu
)



# Dictionary: Models dictionary
MODELS_DICT = {

}