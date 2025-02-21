import torch.nn as nn
import torch
from collections import namedtuple
from typing import List, Any

from utils.datasets import read_platform

def create_model(model_config: namedtuple, classes: List[str], meta_file: str) -> ShipClassificationModel:
    """
    Creates and loads the model with pre-trained weights based on the platform.

    Args:
        model_config (namedtuple): Configuration for the model architecture.
        classes (List[str]): List of class labels.
        meta_file (str): Path to the metadata file.

    Returns:
        ShipClsModel: The initialized and pre-trained ship classification model.
    """
    OUTPUT_DIM = len(classes)
    model = ShipClassificationModel(model_config, OUTPUT_DIM)
    
    platform = read_platform(meta_file)
    if platform not in ['ICEYE', 'K5']:  # 모델 없는 플랫폼 config file에 지정
        platform = 'GENERAL'  # 모델 없는 경우 일반화된 모델 적용 
    
    model.load_state_dict(torch.load(f'./models/weights/{platform}.pt'))
    model.eval()
    return model


def select_model(classes: List[str], meta_file: str) -> ShipClsModel:
    """
    Selects and creates the pre-trained ship classification model.

    Args:
        classes (List[str]): List of class labels.
        meta_file (str): Path to the metadata file.

    Returns:
        ShipClsModel: The pre-trained ship classification model.
    """
    ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])
    model_config = ResNetConfig(
        block=Bottleneck,
        n_blocks=[3, 8, 36, 3],
        channels=[64, 128, 256, 512]
    )
    
    pretrained_model = create_model(model_config, classes, meta_file)
    
    return pretrained_model


class ShipClassificationModel(nn.Module):
    def __init__(self, config: tuple, output_dim: int) -> None:
        """
        Initializes the Ship Classification Model.

        Args:
            config (tuple): A tuple containing the block type, number of blocks, and channels.
            output_dim (int): The number of output classes.
        """
        super().__init__()
                
        block, n_blocks, channels = config
        self.in_channels = channels[0]
            
        assert len(n_blocks) == len(channels) == 4
        
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self.get_resnet_layer(block, n_blocks[0], channels[0])
        self.layer2 = self.get_resnet_layer(block, n_blocks[1], channels[1], stride=2)
        self.layer3 = self.get_resnet_layer(block, n_blocks[2], channels[2], stride=2)
        self.layer4 = self.get_resnet_layer(block, n_blocks[3], channels[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.in_channels, output_dim)
        
    def get_resnet_layer(self, block: nn.Module, n_blocks: int, channels: int, stride: int = 1) -> nn.Sequential:
        """
        Creates a ResNet layer.

        Args:
            block (nn.Module): The block type to use.
            n_blocks (int): The number of blocks.
            channels (int): The number of channels.
            stride (int, optional): The stride for the blocks. Defaults to 1.

        Returns:
            nn.Sequential: A sequential container of the ResNet layer.
        """
        layers = []
        
        downsample = self.in_channels != block.expansion * channels
        layers.append(block(self.in_channels, channels, stride, downsample))
        
        for _ in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))

        self.in_channels = block.expansion * channels
            
        return nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: Output tensor and the flattened features.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.fc(h)
        
        return x, h


class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: bool = False) -> None:
        """
        Initializes the Bottleneck block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int, optional): Stride for the convolutional layer. Defaults to 1.
            downsample (bool, optional): Whether to downsample the input. Defaults to False.
        """
        super().__init__()
    
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        if downsample:
            conv = nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False)
            bn = nn.BatchNorm2d(self.expansion * out_channels)
            self.downsample = nn.Sequential(conv, bn)
        else:
            self.downsample = None
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Bottleneck block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        identity = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
                
        if self.downsample is not None:
            identity = self.downsample(identity)
            
        x += identity
        x = self.relu(x)
    
        return x
