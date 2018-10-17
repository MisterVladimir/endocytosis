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
from .bottleneck import (UpBottleneck, DownBottleneck, ConvMiniBlock)


class ResNetBottom(nn.Module):
    """
    Builds the first block of ResNet.
    """
    def __init__(self, inplanes=1, outplanes=128):
        super().__init__()
        self.block1 = ConvMiniBlock(inplanes, outplanes, relu=True,
                                    kernel_size=7, stride=2, padding=3)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                    ceil_mode=True)

    def forward(self, x):
        x = self.block1(x)
        x = self.maxpool(x)
        return x


# resnet_to_layers = {50: [3, 4, 6, 3],
#                     101: [3, 4, 23, 3],
#                     152: [3, 8, 36, 3]}
class ResNetMiddle(nn.Module):
    """
    Builds the body of ResNet.
    """
    def __init__(self, block, nlayers, inplanes, stride, multiplier=2):
        super().__init__()
        # inplanes = CONFIG.TRAIN.INITIAL_PLANES
        # multiplier = CONFIG.TRAIN.DEPTH_MULTIPLIER
        self.outplanes = inplanes
        if not np.iterable(multiplier):
            multiplier = [multiplier**i for i in len(nlayers)]

        self.layer1 = self._make_layer(
            block, inplanes, nlayers[0], stride=stride[0])
        self.layer2 = self._make_layer(
            block, inplanes * multiplier[1], nlayers[1], stride=stride[1])
        self.layer3 = self._make_layer(
            block, inplanes * multiplier[2], nlayers[2], stride=stride[2])
        if len(nlayers) > 3:
            self.layer4 = self._make_layer(
                block, inplanes * multiplier[3], nlayers[3], stride=stride[3])

    def _make_layer(self, block, inplanes, nblocks, kernel_size=3,
                    stride=1, padding=1):
        layers = []

        if stride > 1 or not self.outplanes == inplanes:
            ds = block.make_sampler(
                self.outplanes, inplanes, kernel_size, stride, padding)
            layers.append(block(sampler=ds))
            nblocks -= 1

        self.outplanes = inplanes
        for _ in range(nblocks):
            layers.append(block(
                inplanes=self.outplanes, kernel_size=kernel_size,
                stride=stride, padding=padding))

        return nn.Sequential(*layers)

    def forward(self, x):
        try:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        except AttributeError:
            return x

        return x


class ResNetTop(nn.Module):
    def __init__(self, inplanes):
        super().__init__()
        self.conv = nn.Conv2d(inplanes, 3, 1)
        self.clamp = nn.Hardtanh(0, 1, True)

    def forward(self, x):
        x = self.conv(x)
        return self.clamp(x)
