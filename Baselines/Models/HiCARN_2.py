# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# HiCARN: https://github.com/OluwadareLab/HiCARN
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


class Basic_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

    def forward(self, x):
        """
        Applies a 1x1, 2D convolutional layer to the input image.
        Args: Matrix of dim=4
        """
        out = self.conv(x)

        return out


class Residual_Block(nn.Module):
    def __init__(self, num_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        """
        Applies the following sequence of functions to the input image:
            3x3 Conv --> Relu --> 3x3 Conv + input --> Relu.
        Args: Matrix of dim=4
        """
        out = F.relu(self.conv1(x), inplace=True)
        out = F.relu(self.conv2(out) + x, inplace=True)

        return out


class Cascading_Block(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.r1 = Residual_Block(channels)
        self.r2 = Residual_Block(channels)
        self.r3 = Residual_Block(channels)
        self.c1 = Basic_Block(channels * 2, channels)
        self.c2 = Basic_Block(channels * 3, channels)
        self.c3 = Basic_Block(channels * 4, channels)

    def forward(self, x):
        """
        Applies the following sequence of functions to the input 3 times:
            Residual Block --> Add previous block input --> 1x1 Conv.
        """
        c0 = o0 = x

        b1 = self.r1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.r2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.r3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3


class Generator(nn.Module):
    def __init__(self, num_channels, input_channels=1, out_channels=None):
        super().__init__()

        # Entry 3x3 convolution layer
        if not out_channels:
            out_channels = input_channels
        self.entry = nn.Conv2d(input_channels, num_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # Cascading blocks
        self.cb1 = Cascading_Block(num_channels)
        self.cb2 = Cascading_Block(num_channels)
        self.cb3 = Cascading_Block(num_channels)
        self.cb4 = Cascading_Block(num_channels)
        self.cb5 = Cascading_Block(num_channels)

        # Body 1x1 convolution layers
        self.cv1 = nn.Conv2d(num_channels * 2, num_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.cv2 = nn.Conv2d(num_channels * 3, num_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.cv3 = nn.Conv2d(num_channels * 4, num_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.cv4 = nn.Conv2d(num_channels * 5, num_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.cv5 = nn.Conv2d(num_channels * 6, num_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        # 3x3 exit convolution layer
        self.exit = nn.Conv2d(num_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        """
        Applies the following sequence of functions to the input:
           3x3 entry Conv --> 3 * (Cascading Block --> Add previous block input --> 1x1 Conv) --> 3x3 exit Conv.
        """
        x = self.entry(x)
        c0 = o0 = x
        b1 = self.cb1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.cv1(c1)

        b2 = self.cb2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.cv2(c2)

        b3 = self.cb3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.cv3(c3)

        b4 = self.cb4(o3)
        c4 = torch.cat([c3, b4], dim=1)
        o4 = self.cv4(c4)

        b5 = self.cb5(o4)
        c5 = torch.cat([c4, b5], dim=1)
        o5 = self.cv5(c5)

        out = self.exit(o5)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()

        def conv_bn_lrelu(in_channels, out_channels, ksize, stride, pad):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU()
            )

        self.body = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.LeakyReLU(),
            conv_bn_lrelu(64, 64, 4, 2, 1),
            conv_bn_lrelu(64, 128, 3, 1, 1),
            conv_bn_lrelu(128, 128, 4, 2, 1),
            conv_bn_lrelu(128, 256, 3, 1, 1),
            conv_bn_lrelu(256, 256, 4, 2, 1),
            conv_bn_lrelu(256, 512, 3, 1, 1),
            conv_bn_lrelu(512, 512, 3, 1, 1),
            nn.Conv2d(512, input_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.body(x)
        return out


class Multi_Discriminator(nn.Module):
    def __init__(self, in_channel=1, out_channel=1):
        super().__init__()

        def conv_bn_lrelu(in_channels, out_channels, ksize, stride, pad):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU()
            )

        self.body = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.LeakyReLU(),
            conv_bn_lrelu(64, 64, 4, 2, 1),
            conv_bn_lrelu(64, 128, 3, 1, 1),
            conv_bn_lrelu(128, 128, 4, 2, 1),
            conv_bn_lrelu(128, 256, 3, 1, 1),
            conv_bn_lrelu(256, 256, 4, 2, 1),
            conv_bn_lrelu(256, 512, 3, 1, 1),
            conv_bn_lrelu(512, 512, 3, 1, 1),
            nn.Conv2d(512, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.body(x)
        return out