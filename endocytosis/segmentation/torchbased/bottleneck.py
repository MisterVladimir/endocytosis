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
from abc import abstractclassmethod

from ...config import CONFIG


class _MiniBlockBase(nn.Module):
    """

    """
    def __init__(self, inplanes, outplanes, relu, kernel_size,
                 stride, padding):
        super().__init__()
        args = [inplanes, outplanes, kernel_size, stride, padding]
        outplanes = int(outplanes)
        self.bn = nn.BatchNorm2d(outplanes)
        if relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

        self.inplanes = inplanes
        self.outplanes = outplanes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = False

    def initialize(self):
        # initialize weights
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    @classmethod
    def calculate_padding(cls, shape, kernel_size, stride):
        # get padding such that
        # [height_in, width_in] == [height_out, width_out] // stride
        divides_evenly = (shape - kernel_size) % stride == 0
        # whether the kernel "stamps" over entire image
        covers_all = kernel_size > stride
        if not (divides_evenly and covers_all):
            raise TypeError('Using image size {}, '.format(shape) +
                            'kernel_size {} '.format(kernel_size) +
                            'and stride {} results in '.format(stride) +
                            'non-integer padding.')
        else:
            return kernel_size % 2

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x


class ConvMiniBlock(_MiniBlockBase):
    """
    Simple module containing one 2d convolution and one batch normalization
    layer. Also stores information about the convolution's kernel size,
    stride, and padding as instance attributes.
    """
    def __init__(self, inplanes, outplanes, relu=False, kernel_size=1,
                 stride=1, padding=0, **kwargs):
        super().__init__(
            inplanes, outplanes, relu, kernel_size, stride, padding)

        self.conv = nn.Conv2d(
            inplanes, outplanes, kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False, **kwargs)

        self.initialize()


class TransposeConvMiniBlock(_MiniBlockBase):
    def __init__(self, inplanes, outplanes, relu=False, kernel_size=1,
                 stride=1, padding=0, **kwargs):
        super().__init__(
            inplanes, outplanes, relu, kernel_size, stride, padding)

        self.conv = nn.ConvTranspose2d(
            inplanes, outplanes, kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False, **kwargs)

        self.initialize()


class _BottleneckBase(nn.Module):
    """
    Abstract base class for a modified version of ResNet's Bottleneck module.
    This class only creates subblock2 and subblock3, which perform the second
    and third convolutions of the Bottleneck. That is, subblock2 performs
    3x3 convolutions on the data that's been downsampled along the channel
    dimension, and subblock3 re-expands the channels with 1x1 convolutions.

    Parameters
    ------------
    innerplanes: int
    Number of planes expected from subblock1.

    outplanes: int
    Number of planes in the output.
    """
    expansion = 4

    def __init__(self, innerplanes, outplanes):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.subblock2 = ConvMiniBlock(
            innerplanes, innerplanes, relu=True, kernel_size=3, stride=1,
            padding=1)

        self.subblock3 = ConvMiniBlock(
            innerplanes, outplanes, relu=False, kernel_size=1,
            stride=1, padding=0)

    def _get_parameters(self, sampler, **kwargs):
        # default arguments
        _kwargs = {'inplanes': None, 'outplanes': None,
                   'kernel_size': 1, 'stride': 1, 'padding': 0}

        if sampler:
            _kwargs = {k: getattr(sampler, k) for k in _kwargs}
            print(_kwargs)
        else:
            _kwargs.update(kwargs)
            if _kwargs['outplanes'] is None:
                _kwargs.update({'outplanes': kwargs['inplanes']})

        innerplanes = _kwargs['outplanes'] // self.expansion
        _kwargs.update({'innerplanes': innerplanes})

        print(_kwargs)

        return _kwargs

    def forward(self, x):
        residual = x

        out = self.subblock1(x)
        out = self.subblock2(out)
        out = self.subblock3(out)

        if self.sampler is not None:
            residual = self.sampler(x)

        out += residual
        out = self.relu(out)

        return out


class DownBottleneck(_BottleneckBase):
    def __init__(self, inplanes=None, outplanes=None, sampler=None, **kwargs):
        kwargs = self._get_parameters(sampler, inplanes=inplanes,
                                      outplanes=outplanes, **kwargs)
        super().__init__(kwargs['innerplanes'], kwargs['outplanes'])
        self.sampler = sampler

        # relu=True, i.e. add ReLU layer
        self.subblock1 = ConvMiniBlock(
            kwargs['inplanes'], kwargs['innerplanes'], True,
            kwargs['kernel_size'], kwargs['stride'], kwargs['padding'])

    @classmethod
    def make_sampler(cls, inplanes, outplanes, kernel_size=3, stride=2, padding=1):
        # kernel_size = 1
        # relu = False
        return ConvMiniBlock(
            inplanes, outplanes, False, kernel_size, stride, padding=padding)


class UpBottleneck(_BottleneckBase):
    def __init__(self, inplanes=None, outplanes=None, sampler=None, **kwargs):
        kwargs = self._get_parameters(sampler, inplanes=inplanes,
                                      outplanes=outplanes, **kwargs)
        super().__init__(kwargs['innerplanes'], kwargs['outplanes'])
        self.sampler = sampler

        # relu=True, i.e. add ReLU layer
        self.subblock1 = TransposeConvMiniBlock(
            kwargs['inplanes'], kwargs['innerplanes'], True,
            kwargs['kernel_size'], kwargs['stride'], kwargs['padding'])

    @classmethod
    def make_sampler(cls, inplanes, outplanes, kernel_size=3, stride=2, padding=1):
        return TransposeConvMiniBlock(
            inplanes, outplanes, False, kernel_size, stride, padding)
