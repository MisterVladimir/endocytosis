# -*- coding: utf-8 -*-
"""
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

from ....config import CONFIG
from .. import resnet
from ..bottleneck import (DownBottleneck, UpBottleneck)
from .loss import CombinedLoss


class SimulatedModel(nn.Module):
    def __init__(self, resnet_model):
        super().__init__()
        self.outchannels = None
        self._setup_bottom()
        self._setup_middle(resnet_model)

    def _setup_bottom(self):
        outchannels = CONFIG.TRAIN.BOTTOM_OUT_PLANES
        ciz = CONFIG.SIMULATED.CROPPED_IMAGE_SIZE
        if len(ciz) == 3:
            inchannels = ciz[0]
        else:
            inchannels = 1

        self.bottom = resnet.ResNetBottom(inchannels, outchannels)
        self.outchannels = self.bottom.outchannels

    def _setup_middle(self, model):
        # block, nlayers, inchannels, stride, multiplier=2
        down = CONFIG.TRAIN.RESNET[model]['down']
        up = CONFIG.TRAIN.RESNET[model]['up']
        inchannels = self.bottom.outchannels
        self.outchannels, self.middle = resnet.build_resnet_middle(
            inchannels, down, up)

    def forward(self, im):
        # print('imshape: {}'.format(im.size()))
        im = self.bottom(im)
        # print('imshape: {}'.format(im.size()))
        im = self.middle(im)
        # print('imshape: {}'.format(im.size()))
        return im
