# -*- coding: utf-8 -*-
"""
This code borrows ideas from faster-rcnn.pytorch, which is under the
MIT license. Its copyright is:

Copyright (c) 2017 Jianwei Yang

@author: Vladimir Shteyn
@email: vladimir.shteyn@googlemail.com

Copyright Vladimir Shteyn, 2018

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import torch
from torch import nn
import numpy as np

from ...config import CONFIG
from .bottleneck import (UpBottleneck, DownBottleneck)


class ResNetBottom(nn.Module):
    """
    Builds the first block of ResNet.
    """
    def __init__(self):
        super().__init__()
        outlayers = CONFIG.SIMULATED.TRAIN.INITIAL_PLANES
        ciz = CONFIG.SIMULATED.TRAIN.CROPPED_IMAGE_SIZE
        if len(ciz) == 3:
            nchannels = ciz[0]
        else:
            nchannels = 1
        self.conv1 = nn.Conv2d(nchannels, outlayers, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(outlayers)
        self.relu = nn.ReLU(inplace=True)
        # change
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0,
                                    ceil_mode=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


# resnet_to_layers = {50: [3, 4, 6, 3],
#                     101: [3, 4, 23, 3],
#                     152: [3, 8, 36, 3]}
class ResNetMiddle(nn.Module):
    """
    Builds the body of ResNet.
    """
    def __init__(self, downlayers, uplayers):
        super().__init__()
        inplanes = CONFIG.SIMULATED.TRAIN.INITIAL_PLANES
        self.inplanes = CONFIG.SIMULATED.TRAIN.INITIAL_PLANES
        multiplier = CONFIG.SIMULATED.TRAIN.DEPTH_MULTIPLIER
        self.layer1d = self._make_dlayer(DownBottleneck, inplanes, downlayers[0])
        self.layer2d = self._make_dlayer(
            DownBottleneck, inplanes * multiplier, downlayers[1], stride=2)
        self.layer3d = self._make_dlayer(
            DownBottleneck, inplanes * multiplier**2, downlayers[2], stride=2)

        if len(downlayers) > 3:
            self.layer4d = self._make_dlayer(DownBottleneck, inplanes * multiplier**3, downlayers[3], stride=2)
            # it is slightly better whereas slower to set stride = 1
            # self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
            self.layer4u = self._make_ulayer()

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, nblocks, stride=1):
        layers = []

        if stride > 1 or not self.inplanes == planes:
            ds = block.make_sampler(self.inplanes, planes, stride)
            layers.append(block(downsample=ds))
            nblocks -= 1

        self.inplanes = planes
        for _ in range(nblocks):
            layers.append(block(inplanes=self.inplanes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
