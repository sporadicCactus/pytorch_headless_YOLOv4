import torch
import torch.nn as nn
import torch.nn.functional as F

from .CSPDarknet53 import _BuildCSPDarknet53
from .SPP_PANet import SpatialPyramidPooling, PANet, Conv


class CSPDarknet53_SPP_PANet(nn.Module):
    def __init__(self):
        super(CSPDarknet53_SPP_PANet, self).__init__()

        # CSPDarknet53 backbone
        self.backbone, feature_channels = _BuildCSPDarknet53()

        # head conv
        self.head_conv = nn.Sequential(
            Conv(feature_channels[-1], feature_channels[-1]//2, 1),
            Conv(feature_channels[-1]//2, feature_channels[-1], 3),
            Conv(feature_channels[-1], feature_channels[-1]//2, 1),
        )

        # Spatial Pyramid Pooling
        self.spp = SpatialPyramidPooling()

        # Path Aggregation Net
        self.panet = PANet(feature_channels)

    def forward(self, x):
        features = list(self.backbone(x))
        features[-1] = self.head_conv(features[-1])
        features[-1] = self.spp(features[-1])
        features = self.panet(features)
        
        return features
