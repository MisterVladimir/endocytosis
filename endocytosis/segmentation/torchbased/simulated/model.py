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


def save(filename, model):
    """
    Save the model's state_dict.
    """
    torch.save(model.state_dict(), filename)


def load(filename, mclass, *args, **kwargs):
    """
    Load a model from pickled state_dict.

    Parameters
    ------------
    filename : str

    mclass : type
        Model class.

    args, kwargs
        Parameters for mclass' constructor.
    """
    model = mclass(*args, **kwargs)
    model.load_state_dict(torch.load(filename))
    return model


class SimulatedModel(nn.Module):
    def __init__(self, resnet_model):
        super().__init__()
        self.resnet_model
        self.outchannels = None
        self._setup_bottom()
        self._setup_middle(resnet_model)

    def _setup_bottom(self):
        outchannels = CONFIG.RESNET.BOTTOM_OUT_PLANES
        ciz = CONFIG.SIMULATED.DATA.CROPPED_IMAGE_SIZE
        if len(ciz) == 3:
            inchannels = ciz[0]
            self.cropped_image_size = np.array(ciz[-2:], dtype=np.uint16)
        else:
            inchannels = 1
            self.cropped_image_size = np.array(ciz, dtype=np.uint16)

        self.bottom = resnet.ResNetBottom(inchannels, outchannels)
        self.outchannels = self.bottom.outchannels

    def _setup_middle(self, model):
        # block, nlayers, inchannels, stride, multiplier=2
        down = CONFIG.RESNET[model]['down']
        up = CONFIG.RESNET[model]['up']
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
