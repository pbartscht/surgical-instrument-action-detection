import torch
import torch.nn as nn
from torchvision.models import resnet50
from timm import create_model  # For EfficientNet

class BaseBackbone(nn.Module):
    """Base class for all backbone architectures"""
    def __init__(self):
        super().__init__()
        self.output_channels = None
    
    def get_output_channels(self):
        return self.output_channels

class ResNet50Backbone(BaseBackbone):
    """ResNet50 backbone implementation"""
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = resnet50(pretrained=pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.output_channels = 2048  # ResNet50's output channels
        
    def forward(self, x):
        return self.features(x)

class EfficientNetT4Backbone(BaseBackbone):
    """EfficientNet-T4 backbone implementation"""
    def __init__(self, pretrained=True):
        super().__init__()
        # Load EfficientNet-T4 without classifier
        self.model = create_model(
            'efficientnet_b4',
            pretrained=pretrained,
            features_only=True,
            out_indices=[4]  # Use the last feature map
        )
        self.output_channels = 1792  # EfficientNet-B4's output channels
        
    def forward(self, x):
        features = self.model(x)
        return features[-1]  # Return last feature map
class ConvNeXtBackbone(BaseBackbone):
    """ConvNeXt backbone optimized for surgical instrument detection"""
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = create_model(
            'convnext_base',
            pretrained=pretrained,
            features_only=True,
            out_indices=[4]
        )
        self.output_channels = 1024  # ConvNeXt base output channels
        
    def forward(self, x):
        features = self.model(x)
        return features[-1]

class SwinTransformerBackbone(BaseBackbone):
    """Swin Transformer backbone for capturing spatial relationships"""
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = create_model(
            'swin_base_patch4_window7_224',
            pretrained=pretrained,
            features_only=True,
            out_indices=[4]
        )
        self.output_channels = 1024  # Swin base output channels
        
    def forward(self, x):
        features = self.model(x)
        return features[-1]

class DenseNetBackbone(BaseBackbone):
    """DenseNet backbone for fine-grained instrument movement detection"""
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = create_model(
            'densenet201',
            pretrained=pretrained,
            features_only=True,
            out_indices=[4]
        )
        self.output_channels = 1920  # DenseNet201 output channels
        
    def forward(self, x):
        features = self.model(x)
        return features[-1]

def get_backbone(name='resnet50', **kwargs):
    """
    Backbone factory function
    
    Args:
        name (str): Name of the backbone architecture
        **kwargs: Additional arguments for the backbone
        
    Returns:
        BaseBackbone: Instantiated backbone model
    """
    backbones = {
        'resnet50': ResNet50Backbone,
        'efficientnet_b4': EfficientNetT4Backbone,
        'convnext_base': ConvNeXtBackbone,
        'swin_base': SwinTransformerBackbone,
        'densenet201': DenseNetBackbone,
    }
    
    if name not in backbones:
        raise ValueError(f"Backbone {name} not found. Available backbones: {list(backbones.keys())}")
    
    return backbones[name](**kwargs)