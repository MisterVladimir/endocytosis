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
        self._setup_bottom()
        self._setup_middle(resnet_model)
        self._setup_loss()

    def _setup_bottom(self):
        outplanes = CONFIG.TRAIN.BOTTOM_OUT_PLANES
        ciz = CONFIG.SIMULATED.CROPPED_IMAGE_SIZE
        if len(ciz) == 3:
            nchannels = ciz[0]
        else:
            nchannels = 1

        self.bottom = resnet.ResNetBottom(nchannels, outplanes)

    def _setup_middle(self, model):
        downlayers = CONFIG.TRAIN.RESNET[model]['DOWNLAYERS']
        uplayers = CONFIG.TRAIN.RESNET[model]['UPLAYERS']

        stride = CONFIG.TRAIN.RESNET.STRIDE
        inplanes = CONFIG.TRAIN.BOTTOM_OUT_PLANES

        self.down = resnet.ResNetMiddle(
            DownBottleneck, downlayers, inplanes, 2, stride)

        inplanes = self.down.outplanes
        self.up = resnet.ResNetMiddle(
            UpBottleneck, uplayers, inplanes, 0.5, stride)

        self.outplanes = self.up.outplanes

    def forward(self, im):
        im = self.bottom(im)
        im = self.down(im)
        im = self.up(im)
        return im
