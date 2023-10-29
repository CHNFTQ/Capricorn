# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# HiCARN: https://github.com/OluwadareLab/HiCARN
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg16

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
    def __init__(self, num_channels, input_channels, out_channels=None):
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



class GeneratorLoss(nn.Module):
    def __init__(self, input_channels=1, weight=None):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        self.weight = weight
        if weight:
            weight = torch.tensor(weight)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()
        self.input_channels=input_channels

    def forward(self, out_images, target_images):
        # Perception Loss
        bsz, _, w, h = out_images.shape
        out_feat = self.loss_network(out_images.reshape(-1, w, h).unsqueeze(1).repeat([1, 3, 1, 1]))
        target_feat = self.loss_network(target_images.reshape(-1, w, h).unsqueeze(1).repeat([1, 3, 1, 1]))
        perception_loss = self.mse_loss(out_feat.reshape(out_feat.size(0), -1),
                                        target_feat.reshape(target_feat.size(0), -1))
        # Image Loss
        if self.weight:
            image_loss = self.mse_loss(out_images, target_images)
        else:
            w = self.weight.view(1, -1, 1, 1)
            image_loss = self.mse_loss(out_images * w, target_images * w)
        tv_loss = self.tv_loss(out_images)
        return image_loss + 0.001 * perception_loss + 2e-8 * tv_loss


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        b, c, h, w = x.shape
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w-1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / b

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]
