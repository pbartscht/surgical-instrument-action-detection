from .backbones import (
    BaseBackbone,
    ResNet50Backbone,
    EfficientNetT4Backbone,
    ConvNeXtBackbone,      # Gut für feine Details
    SwinTransformerBackbone,  # Gut für räumliche Beziehungen
    DenseNetBackbone,      # Gut für Feature-Wiederverwendung
    get_backbone
)

# Dictionary mit verfügbaren Backbones und deren Beschreibungen
AVAILABLE_BACKBONES = {
    'resnet50': 'Standard ResNet50 backbone with good general performance',
    'efficientnet_b4': 'Efficient architecture with good accuracy/computation trade-off',
    'convnext_base': 'Modern architecture good at fine-grained instrument details',
    'swin_base': 'Transformer-based model good at capturing spatial relationships',
    'densenet201': 'Dense connections for better feature propagation',
}

# Standard-Konfigurationen für verschiedene Anwendungsfälle
BACKBONE_CONFIGS = {
    'high_precision': {
        'backbone': 'convnext_base',
        'pretrained': True,
        'freeze_layers': 0.7  # Freeze first 70% of layers
    },
    'fast_inference': {
        'backbone': 'efficientnet_b4',
        'pretrained': True,
        'freeze_layers': 0.8
    },
    'balanced': {
        'backbone': 'resnet50',
        'pretrained': True,
        'freeze_layers': 0.6
    }
}

__all__ = [
    'BaseBackbone',
    'ResNet50Backbone',
    'EfficientNetT4Backbone',
    'ConvNeXtBackbone',
    'SwinTransformerBackbone',
    'DenseNetBackbone',
    'get_backbone',
    'AVAILABLE_BACKBONES',
    'BACKBONE_CONFIGS'
]