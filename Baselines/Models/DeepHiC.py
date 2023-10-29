# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeepHiC: https://github.com/omegahh/DeepHiC
# --------------------------------------------------------
import torch
import torch.nn as nn


def swish(x):
    return x * torch.sigmoid(x)


# DeepHiC Residual Block
class residualBlock(nn.Module):
    def __init__(self, channels, k=3, s=1):
        super(residualBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, k, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        # a swish layer here
        self.conv2 = nn.Conv2d(channels, channels, k, stride=s, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = swish(self.bn1(self.conv1(x)))
        residual = self.bn2(self.conv2(residual))
        return x + residual


# DeepHiC Generator
class Generator(nn.Module):
    def __init__(self, num_channels=64, resblock_num=5):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(1, num_channels, kernel_size=9, stride=1, padding=4)
        # have a swish here in forward

        resblocks = [residualBlock(num_channels) for _ in range(resblock_num)]
        self.resblocks = nn.Sequential(*resblocks)

        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        # have a swish here in forward

        self.conv3 = nn.Conv2d(num_channels, 1, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        emb = swish(self.conv1(x))
        x = self.resblocks(emb)
        x = swish(self.bn2(self.conv2(x)))
        x = self.conv3(x + emb)
        return (torch.tanh(x) + 1) / 2


# DeepHiC Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channel=3):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, 3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        # Replaced original paper FC layers with FCN
        self.conv7 = nn.Conv2d(256, 1, 1, stride=1, padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size = x.size(0)

        x = swish(self.conv1(x))
        x = swish(self.bn2(self.conv2(x)))
        x = swish(self.bn3(self.conv3(x)))
        x = swish(self.bn4(self.conv4(x)))
        x = swish(self.bn5(self.conv5(x)))
        x = swish(self.bn6(self.conv6(x)))

        x = self.conv7(x)
        x = self.avgpool(x)
        return torch.sigmoid(x.view(batch_size))
