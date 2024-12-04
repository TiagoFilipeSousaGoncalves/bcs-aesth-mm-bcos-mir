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
    bcosconvnext_atto,
    bcosconvnext_tiny,
    bcosconvnext_small,
    bcosconvnext_base,
    bcosconvnext_large
)
from bcos.models.densenet import (
    baseline_densenet121,
    baseline_densenet161,
    baseline_densenet169,
    baseline_densenet201,
    bcosdensenet121,
    bcosdensenet161,
    bcosdensenet169,
    bcosdensenet201,
)
from bcos.models.pretrained import (
    bcos_pretr_convnext_base,
    bcos_pretr_vgg11_bnu,
    bcos_pretr_resnext50_32x4d,
    bcos_pretr_convnext_base_bnu,
    bcos_pretr_convnext_tiny,
    bcos_pretr_convnext_tiny_bnu,
    bcos_pretr_densenet121,
    bcos_pretr_densenet121_long,
    bcos_pretr_densenet161,
    bcos_pretr_densenet169,
    bcos_pretr_densenet201,
    bcos_pretr_resnet101,
    bcos_pretr_resnet152,
    bcos_pretr_resnet152_long,
    bcos_pretr_resnet18,
    bcos_pretr_resnet34,
    bcos_pretr_resnet50
)
from bcos.models.resnet import (
    bcosresnet18,
    bcosresnet34,
    bcosresnet50,
    bcosresnet101,
    bcosresnet152,
    bcosresnext50_32x4d,
    bcosresnext101_32x8d,
    bcoswide_resnet50_2,
    bcoswide_resnet101_2
)
from bcos.models.vgg import (
    bcosvgg11,
    bcosvgg11_bnu,
    bcosvgg13,
    bcosvgg13_bnu,
    bcosvgg16,
    bcosvgg16_bnu,
    bcosvgg19,
    bcosvgg19_bnu
)



# Dictionary: Models dictionary
MODELS_DICT = {
    "baseline_densenet121":baseline_densenet121(num_classes=0, pretrained=False),
    "baseline_densenet161":baseline_densenet161(num_classes=0, pretrained=False),
    "baseline_densenet169":baseline_densenet169(num_classes=0, pretrained=False),
    "baseline_densenet201":baseline_densenet201(num_classes=0, pretrained=False),
    "bcosdensenet121":bcosdensenet121({'num_classes':0}),
    "bcosdensenet161":bcosdensenet161({'num_classes':0}),
    "bcosdensenet169":bcosdensenet169({'num_classes':0}),
    "bcosdensenet201":bcosdensenet201({'num_classes':0})
}