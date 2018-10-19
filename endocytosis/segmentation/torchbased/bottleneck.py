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
import copy

from ...config import CONFIG


class _MiniBlockBase(nn.Module):
    """

    """
    def __init__(self, inchannels, outchannels, relu, kernel_size,
                 stride, padding):
        super().__init__()
        outchannels = int(outchannels)
        self.bn = nn.BatchNorm2d(outchannels)
        if relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

        self.inchannels = inchannels
        self.outchannels = outchannels
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
        print('start size: {}'.format(x.size()))
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        print('end size: {}'.format(x.size()))
        return x


class ConvMiniBlock(_MiniBlockBase):
    """
    Simple module containing one 2d convolution and one batch normalization
    layer. Also stores information about the convolution's kernel size,
    stride, and padding as instance attributes.
    """
    def __init__(self, inchannels, outchannels, relu=False, kernel_size=1,
                 stride=1, padding=0, **kwargs):
        super().__init__(
            inchannels, outchannels, relu, kernel_size, stride, padding)

        self.conv = nn.Conv2d(
            inchannels, outchannels, kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False)

        self.initialize()


class TransposeConvMiniBlock(_MiniBlockBase):
    def __init__(self, inchannels, outchannels, relu, kernel_size,
                 stride, padding, output_padding, **kwargs):
        super().__init__(
            inchannels, outchannels, relu, kernel_size, stride, padding)

        self.output_padding = output_padding

        self.conv = nn.ConvTranspose2d(
            inchannels, outchannels, kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False, output_padding=output_padding)

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
    innerchannels: int
    Number of planes expected from subblock1.

    outchannels: int
    Number of planes in the output.
    """
    expansion = 4
    default_kwargs = {}
    convblock = None

    def __init__(self, inchannels=None, outchannels=None, sampler=None, **kwargs):
        super().__init__()

        kwargs = self._get_parameters(
            sampler, inchannels=inchannels, outchannels=outchannels, **kwargs)
        self.sampler = sampler

        # relu=True, i.e. add ReLU layer
        self.subblock1 = self.convblock(relu=True, **kwargs)

        self.relu = nn.ReLU(inplace=True)

        innerchannels = self.subblock1.outchannels
        self.subblock2 = ConvMiniBlock(
            innerchannels, innerchannels, relu=True, kernel_size=3, stride=1,
            padding=1)

        outchannels = innerchannels * self.expansion
        self.subblock3 = ConvMiniBlock(
            innerchannels, outchannels, relu=False, kernel_size=1,
            stride=1, padding=0)

    def _get_parameters(self, sampler, **kwargs):
        # default arguments
        _kwargs = copy.deepcopy(self.default_kwargs)
        # if we're up- or down-sampling the input, get parameters
        # from the Module performing the up- or down-sampling
        if sampler:
            _kwargs = {k: getattr(sampler, k) for k in _kwargs}
            innerchannels = _kwargs['outchannels'] // self.expansion
            _kwargs.update({'outchannels': innerchannels})
        # otherwise, non-default parameters should be provided in kwargs
        else:
            _kwargs.update(kwargs)
            if _kwargs['outchannels'] is None:
                innerchannels = _kwargs['inchannels'] // self.expansion
                _kwargs.update({'outchannels': innerchannels})

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
    default_kwargs = {'inchannels': None, 'outchannels': None,
                      'kernel_size': 1, 'stride': 1, 'padding': 0}
    convblock = ConvMiniBlock

    @classmethod
    def make_sampler(cls, inchannels, outchannels, kernel_size, stride,
                     padding, **kwargs):

        # relu = False
        return ConvMiniBlock(
            inchannels, outchannels, False, kernel_size, stride, padding)


class UpBottleneck(_BottleneckBase):
    default_kwargs = {'inchannels': None, 'outchannels': None, 'kernel_size': 1,
                      'stride': 1, 'padding': 0, 'output_padding': 0}
    convblock = TransposeConvMiniBlock

    @classmethod
    def make_sampler(cls, inchannels, outchannels, kernel_size, stride,
                     padding, output_padding, **kwargs):

        return TransposeConvMiniBlock(
            inchannels, outchannels, False, kernel_size,
            stride, padding, output_padding)
