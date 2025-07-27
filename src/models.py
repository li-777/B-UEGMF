"""
Neural Network Model Definitions
Contains various network architectures for experiments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Tuple
import logging

from config import Config

logger = logging.getLogger(__name__)


class BasicBlock(nn.Module):
    """ResNet Basic Block"""
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet Network"""
    
    def __init__(self, block, num_blocks: list, num_classes: int = 10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class VGG(nn.Module):
    """VGG Network"""
    
    def __init__(self, cfg: list, num_classes: int = 10, batch_norm: bool = False):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg, batch_norm)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg: list, batch_norm: bool):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1)]
                if batch_norm:
                    layers += [nn.BatchNorm2d(x)]
                layers += [nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class DenseBlock(nn.Module):
    """DenseNet Dense Block"""
    
    def __init__(self, num_layers: int, in_channels: int, growth_rate: int, bn_size: int, drop_rate: float):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(
                in_channels + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)


class DenseLayer(nn.Module):
    """DenseNet Dense Layer"""
    
    def __init__(self, in_channels: int, growth_rate: int, bn_size: int, drop_rate: float):
        super(DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(in_channels)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(in_channels, bn_size * growth_rate,
                                         kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                         kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class TextFeatureExtractor(nn.Module):
    """Text Feature Extractor Network"""
    
    def __init__(self, 
                 input_dim: int = 768,  # BERT output dimension
                 hidden_dim: int = Config.DEM.TEXT_HIDDEN_DIM,
                 num_layers: int = Config.DEM.TEXT_NUM_LAYERS):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim) for _ in range(num_layers)
        ])
        
        # Create different output projection layers for different datasets
        self.output_projections = nn.ModuleDict({
            'cifar10': nn.Linear(hidden_dim, 32*32*3),
            'cifar100': nn.Linear(hidden_dim, 32*32*3),
            'imagenet': nn.Linear(hidden_dim, 224*224*3)
        })
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, text_embeddings: torch.Tensor, target_shape: tuple, dataset_name: str = 'cifar10'):
        """
        Forward pass
        
        Args:
            text_embeddings: BERT text embeddings [batch_size, 768]
            target_shape: Target image shape (batch_size, channels, height, width)
            dataset_name: Dataset name
            
        Returns:
            torch.Tensor: Text perturbation [batch_size, channels, height, width]
        """
        batch_size, channels, height, width = target_shape
        
        # Input projection
        x = self.input_projection(text_embeddings)  # [batch_size, hidden_dim]
        x = self.dropout(x)
        
        # Through Transformer blocks
        x = x.unsqueeze(1)  # [batch_size, 1, hidden_dim] add sequence dimension
        for block in self.transformer_blocks:
            x = block(x)
        x = x.squeeze(1)  # [batch_size, hidden_dim]
        
        # Output projection to target shape
        try:
            if dataset_name in self.output_projections:
                perturbation = self.output_projections[dataset_name](x)
            else:
                # Default to CIFAR projection
                perturbation = self.output_projections['cifar10'](x)
            
            # Reshape to image shape
            expected_size = channels * height * width
            if perturbation.size(1) != expected_size:
                # If output size doesn't match, use linear interpolation to adjust
                perturbation = perturbation[:, :expected_size]  # Truncate
                if perturbation.size(1) < expected_size:
                    # If too small, pad with zeros
                    pad_size = expected_size - perturbation.size(1)
                    padding = torch.zeros(batch_size, pad_size, device=perturbation.device)
                    perturbation = torch.cat([perturbation, padding], dim=1)
            
            perturbation = perturbation.view(batch_size, channels, height, width)
            
        except Exception as e:
            # If reshaping fails, create zero tensor with same shape
            logger.warning(f"Text perturbation reshaping failed: {e}, using zero perturbation")
            perturbation = torch.zeros(batch_size, channels, height, width, device=text_embeddings.device)
        
        # Normalize to perturbation range
        perturbation = torch.tanh(perturbation) * (8/255)  # Use fixed epsilon value
        
        return perturbation


class TransformerBlock(nn.Module):
    """Transformer Block"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-head attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward network
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


def create_model(model_name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """
    Factory function to create models
    
    Args:
        model_name: Model name
        num_classes: Number of classes
        pretrained: Whether to use pretrained weights
        
    Returns:
        nn.Module: Model instance
    """
    model_name = model_name.lower()
    
    if model_name == 'resnet18':
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
    
    elif model_name == 'resnet34':
        return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)
    
    elif model_name == 'resnet50':
        if pretrained:
            model = models.resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            model = models.resnet50(pretrained=False, num_classes=num_classes)
        return model
    
    elif model_name == 'vgg16':
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        return VGG(cfg, num_classes, batch_norm=True)
    
    elif model_name == 'vgg19':
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        return VGG(cfg, num_classes, batch_norm=True)
    
    elif model_name == 'densenet121':
        if pretrained:
            model = models.densenet121(pretrained=True)
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        else:
            model = models.densenet121(pretrained=False, num_classes=num_classes)
        return model
    
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def get_model_info(model: nn.Module) -> dict:
    """
    Get model information
    
    Args:
        model: Model instance
        
    Returns:
        dict: Model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assume float32
    }


def initialize_weights(model: nn.Module) -> None:
    """
    Initialize model weights
    
    Args:
        model: Model instance
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def load_pretrained_model(model_path: str, model: nn.Module, device: str = 'cpu') -> nn.Module:
    """
    Load pretrained model
    
    Args:
        model_path: Model file path
        model: Model instance
        device: Device
        
    Returns:
        nn.Module: Model with loaded weights
    """
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logger.info(f"Successfully loaded pretrained model: {model_path}")
    except Exception as e:
        logger.error(f"Failed to load pretrained model: {e}")
        raise
    
    return model


def save_model(model: nn.Module, 
               model_path: str, 
               epoch: int, 
               optimizer: Optional[torch.optim.Optimizer] = None,
               loss: Optional[float] = None,
               accuracy: Optional[float] = None) -> None:
    """
    Save model
    
    Args:
        model: Model instance
        model_path: Save path
        epoch: Training epoch
        optimizer: Optimizer
        loss: Loss value
        accuracy: Accuracy
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    torch.save(checkpoint, model_path)
    logger.info(f"Model saved: {model_path}")


# Supported model list
SUPPORTED_MODELS = [
    'resnet18', 'resnet34', 'resnet50',
    'vgg16', 'vgg19',
    'densenet121'
]


if __name__ == "__main__":
    # Test model creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Testing model creation...")
    for model_name in ['resnet18', 'vgg16']:
        print(f"\nCreating {model_name}...")
        model = create_model(model_name, num_classes=10)
        model.to(device)
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 32, 32).to(device)
        output = model(dummy_input)
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        
        # Model information
        info = get_model_info(model)
        print(f"  Parameters: {info['total_parameters']:,}")
        print(f"  Model size: {info['model_size_mb']:.2f} MB")
    
    # Test text feature extractor
    print(f"\nTesting text feature extractor...")
    text_extractor = TextFeatureExtractor().to(device)
    dummy_text_emb = torch.randn(2, 768).to(device)  # BERT output
    target_shape = (2, 3, 32, 32)
    perturbation = text_extractor(dummy_text_emb, target_shape)
    print(f"  Text embedding shape: {dummy_text_emb.shape}")
    print(f"  Perturbation shape: {perturbation.shape}")
    print(f"  Perturbation range: [{perturbation.min():.4f}, {perturbation.max():.4f}]")