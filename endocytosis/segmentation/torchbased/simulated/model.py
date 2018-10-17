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

    def _setup_bottom(self):
        outplanes = CONFIG.TRAIN.BOTTOM_OUT_PLANES
        ciz = CONFIG.SIMULATED.CROPPED_IMAGE_SIZE
        if len(ciz) == 3:
            nchannels = ciz[0]
        else:
            nchannels = 1

        self.bottom = resnet.ResNetBottom(nchannels, outplanes)

    def _setup_middle(self, model):
        # block, nlayers, inplanes, stride, multiplier=2
        down = CONFIG.TRAIN.RESNET[model]['DOWN']
        downlayers = down.LAYERS
        downstride = down.STRIDE
        downmult = down.MULTIPLIER
        inplanes = self.bottom.outplanes
        self.down = resnet.ResNetMiddle(
            block=DownBottleneck, nlayers=downlayers, inplanes=inplanes,
            stride=downstride, multiplier=downmult)

        up = CONFIG.TRAIN.RESNET[model]['UP']
        uplayers = up.LAYERS
        upstride = up.STRIDE
        upmult = up.MULTIPLIER
        inplanes = self.down.outplanes
        self.up = resnet.ResNetMiddle(
            block=UpBottleneck, nlayers=uplayers, inplanes=inplanes,
            stride=upstride, multiplier=upmult)

        self.outplanes = self.up.outplanes

    def forward(self, im):
        im = self.bottom(im)
        im = self.down(im)
        im = self.up(im)
        return im
