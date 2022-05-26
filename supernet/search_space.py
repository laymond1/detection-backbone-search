import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ofa.utils.layers import (
    set_layer_from_config,
    MBConvLayer,
)

blocks_key = [
    'mobilenet_3x3',
    'mobilenet_5x5',
    'mobilenet_7x7',
]

Blocks = {
    'mobilenet_3x3': lambda in_channels, out_channels, stride, expand_ratio: MBConvLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, expand_ratio=expand_ratio),
    'mobilenet_5x5': lambda in_channels, out_channels, stride, expand_ratio: MBConvLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=stride, expand_ratio=expand_ratio),
    'mobilenet_7x7': lambda in_channels, out_channels, stride, expand_ratio: MBConvLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=stride, expand_ratio=expand_ratio),
}