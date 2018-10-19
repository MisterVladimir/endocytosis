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
from fijitools.helpers.iteration import isiterable

from ...config import CONFIG
from .bottleneck import (UpBottleneck, DownBottleneck, ConvMiniBlock)


class ResNetBottom(nn.Module):
    """
    Builds the first block of ResNet.
    """
    def __init__(self, inchannels=1, outchannels=128):
        super().__init__()
        self.outchannels = outchannels
        self.block1 = ConvMiniBlock(inchannels, outchannels, relu=True,
                                    kernel_size=5, stride=2, padding=2)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                    ceil_mode=True)

    def forward(self, x):
        x = self.block1(x)
        # x = self.maxpool(x)
        return x


class _ResNetMiddle(nn.Module):
    def __init__(self, inchannels):
        super().__init__()
        self.outchannels = inchannels
        self.layers = nn.ModuleList()

    def add_layer(self, inchannels, nblocks, **kwargs):
        """
        Add nblocks number of skip connection mini neural nets.

        Parameters
        ------------
        nchannels : int
            Number of input channels.

        nblocks : int
            Number of skip connection blocks in this layer. For example
            resnet50 has two such blocks in its first layer.

        stride : int

        kwargs
            Arguments for constructing a convolutional layer. These are fed
            into the up/downsampler and the first block within this layer.
            They are:

            kernel_size : int
            padding : int
            output_padding : int
                Note that this parameter is only valid when self.block is a
                transpose convolution Module.
        """

        layers = []
        # gotcha: if we set stride as a keyword argument with a default value
        # of 1, we fail to pass the stride variable along with **kwargs
        if 'stride' not in kwargs:
            stride = 1
        else:
            stride = kwargs['stride']

        if stride > 1 or not self.outchannels == inchannels:
            ds = self.block.make_sampler(self.outchannels, inchannels, **kwargs)
            layers.append(self.block(sampler=ds))
            nblocks -= 1

        self.outchannels = inchannels
        for _ in range(nblocks):
            layers.append(self.block(
                inchannels=self.outchannels, kernel_size=1,
                stride=1, padding=0))
        # Add to instance variable only at the end in case error crops up
        # while instantiating a new self.block object. This way, errors during
        # for example the above iteration don't leave self.layers dangling with
        # some intermediate number of blocks.
        self.layers.extend(layers)

    def clear_layers(self):
        self.layers = nn.ModuleList()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class UpResNet(_ResNetMiddle):
    block = UpBottleneck


class DownResNet(_ResNetMiddle):
    block = DownBottleneck


def build_resnet_middle(inchannels, down, up=None):
    """
    Build the skip connection layers of ResNet.

    Parameters
    ------------
    inchannels : int
        Number of channels in the input data.

    down : dict
    up : dict
        Keyword arguments to the add_layer methods of DownResNet and UpResNet,
        respectively.


    """
    def filter_dict(dic):
        """
        If a value in the dictionary is not iterable, repeat it 'length'
        times, where 'length' is the minimum length of an iterable in
        the dictionary's values.
        """
        keys = []
        lengths = []
        for k, v in dic.items():
            if isiterable(v):
                lengths.append(len(v))
            else:
                keys.append(k)
        lengths = min(lengths)
        for k in keys:
            dic.update({k: [dic[k]] * lengths})
        return lengths, dic

    def add_layers(net, dic, length):
        for i in range(length):
            kwargs = {k: v[i] for k, v in dic.items()}
            net.add_layer(**kwargs)

    downlength, down = filter_dict(down)
    downres = DownResNet(inchannels)
    # print(down)
    add_layers(downres, down, downlength)
    result = nn.ModuleList([downres, ])
    # add layers to upsampling ResNet
    if up:
        upres = UpResNet(downres.outchannels)
        uplength, down = filter_dict(up)
        add_layers(upres, up, uplength)
        result.append(upres)

    outchannels = result[-1].outchannels
    return outchannels, nn.Sequential(*result)


class ResNetTop(nn.Module):
    def __init__(self, inchannels):
        super().__init__()
        self.conv = nn.Conv2d(inchannels, 3, 1)
        self.clamp = nn.Hardtanh(0, 1, True)

    def forward(self, x):
        x = self.conv(x)
        return self.clamp(x)
