# Imports
from PIL import Image
import timm

# PyTorch Imports
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights, vgg16, VGG16_Weights



# Dictionary: Models dictionary
MODELS_DICT = {
    "Google_Base_Patch16_224":Google_Base_Patch16_224(),
    "DeiT_Base_Patch16_224":DeiT_Base_Patch16_224(),
    "Beit_Base_Patch16_224":Beit_Base_Patch16_224(),
    "DinoV2_Base_Patch16_224":DinoV2_Base_Patch16_224(),
    "ResNet50_Base_224":ResNet50_Base_224(),
    "VGG16_Base_224":VGG16_Base_224(),
    "CrossViT_Tiny240":CrossViT_Tiny240(),
    "LeViTConv256":LeViTConv256(),
    "ConViT_Tiny":ConViT_Tiny(),
    "MaxViT_Tiny_224":MaxViT_Tiny_224(),
    "MViTv2_Tiny":MViTv2_Tiny(),
    "DaViT_Tiny":DaViT_Tiny(),
    "Google_Base_Patch16_224_MLP":Google_Base_Patch16_224_MLP(),
    "DinoV2_Base_Patch16_224_MLP":DinoV2_Base_Patch16_224_MLP(),
    "Beit_Base_Patch16_224_MLP":Beit_Base_Patch16_224_MLP(),
    "DeiT_Base_Patch16_224_MLP":DeiT_Base_Patch16_224_MLP(),
    "ResNet50_Base_224_MLP":ResNet50_Base_224_MLP(),
    "VGG16_Base_224_MLP":VGG16_Base_224_MLP()
}