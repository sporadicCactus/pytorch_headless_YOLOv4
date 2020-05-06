import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(Conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=.1)
        )

    def forward(self, x):
        return self.conv(x)

class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)

        return features

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            Conv(in_channels, out_channels, 1),
        )

    def forward(self, x, target_size):
        x = self.upsample(x)
        x = F.interpolate(x, target_size, mode='bilinear', align_corners=False)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()

        self.downsample = Conv(in_channels, out_channels, 3, 2)

    def forward(self, x):
        return self.downsample(x)

class PANet(nn.Module):
    def __init__(self, feature_channels):
        super(PANet, self).__init__()

        self.initial_transforms = nn.ModuleList([Conv(channels, channels//2, 1) for channels in feature_channels[:-1]] +\
                                                [nn.Sequential(
                                                    Conv(feature_channels[-1]*2, feature_channels[-1]//2, 1),
                                                    Conv(feature_channels[-1]//2, feature_channels[-1], 3),
                                                    Conv(feature_channels[-1], feature_channels[-1]//2, 1))
                                                ])

        self.downstream_transforms = nn.ModuleList([self._horizontal_stack(channels) for channels in feature_channels[:-1]] + [nn.Identity()])
        self.upstream_transforms = nn.ModuleList([nn.Identity()] + [self._horizontal_stack(channels) for channels in feature_channels[1:]])

        self.upsamplers = nn.ModuleList([Upsample(channel_high//2, channel_low//2) for channel_high, channel_low in zip(feature_channels[1:], feature_channels[:-1])])
        self.downsamplers = nn.ModuleList([Downsample(channel_low//2, channel_high//2) for channel_low, channel_high in zip(feature_channels[:-1], feature_channels[1:])])

    def _horizontal_stack(self, channels):
        return nn.Sequential(
            Conv(channels, channels//2, 1),
            Conv(channels//2, channels, 3),
            Conv(channels, channels//2, 1),
            Conv(channels//2, channels, 3),
            Conv(channels, channels//2, 1),
        )

    def forward(self, features):
        features = [tr(f) for tr, f in zip(self.initial_transforms, features)]

        # descending the feature pyramid
        features[-1] = self.downstream_transforms[-1](features[-1])
        for ind in range(len(features) - 1, 0, -1):
            features[ind - 1] = torch.cat([features[ind - 1], self.upsamplers[ind - 1](features[ind], features[ind - 1].shape[-2:])], dim=1)
            features[ind - 1] = self.downstream_transforms[ind - 1](features[ind - 1])


        # ascending the feature pyramid
        features[0] = self.upstream_transforms[0](features[0])
        for ind in range(0, len(features) - 1, +1):
            features[ind + 1] = torch.cat([self.downsamplers[ind](features[ind]), features[ind + 1]], dim=1)
            features[ind + 1] = self.upstream_transforms[ind + 1](features[ind + 1])

        return tuple(features)

