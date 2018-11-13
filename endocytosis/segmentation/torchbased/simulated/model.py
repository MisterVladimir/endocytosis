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
from ..enums import ModelTask


def save(filename, model):
    """
    Save the model's state_dict.
    """
    torch.save(model.state_dict(), filename)
    return True


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


class RPN(nn.Module):
    def __init__(self, inchannels, imshape, training):
        super().__init__()
        if training | (ModelTask.TRAINING | ModelTask.TESTING):
            self.mask_cutoff = CONFIG.SIMULATED.TRAIN.MASK_CUTOFF
        else:
            self.mask_cutoff = CONFIG.SIMULATED.INFER.MASK_CUTOFF

        self.conv = nn.Conv2d(inchannels, inchannels, kernel_size=1)
        self.mask_conv = nn.Conv2d(inchannels, out_channels=1, kernel_size=1)
        self.deltas_conv = nn.Conv2d(inchannels, out_channels=2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def _make_grid(self, shape):
        xc = torch.arange(shape[0], dtype=torch.float32)
        yc = torch.arange(shape[1], dtype=torch.float32)
        return torch.stack(torch.meshgrid([xc, yc]), dim=0)

    def apply_mask(self, p, deltas):
        """
        Parameters
        ------------
        Outputs of self.forward

        p : torch.Tensor (dtype=torch.float32, shape=(1, n, m))
        Each pixel gives the probability that it contains a spot centroid.

        deltas : torch.Tensor (dtype=torch.float32, shape=(2, n, m))
        X and Y coordinates of the spot centroid within that pixel.


        Returns
        ---------
        deltas : torch.Tensor (dtype=torch.float32, shape=(2, r))
        Array of X and Y coordinates whose probability is higher than the
        cutoff probability.
        """
        mask = p > self.mask_cutoff
        deltas = torch.masked_select(deltas, mask).reshape((2, -1))
        return deltas

    def forward(self, x):
        if not hasattr(self, 'grid'):
            self.grid = self._make_grid(x.size()[-2:]).to(x.device)
        deltas = self.conv(x)
        deltas = self.deltas_conv(deltas) + self.grid + 0.5

        p = self.conv(x)
        p = self.mask_conv(p)
        p = self.sigmoid(p[0])
        p = 0.5 * (p + 1.)

        return p, deltas


class SimulationModel(nn.Module):
    def __init__(self, resnet_model, training):
        super().__init__()
        self.training_flag = training
        self.outchannels = None
        self._setup_bottom()
        self._setup_middle(resnet_model)
        self._setup_top(self.cropped_image_size[-2:])

    def _setup_bottom(self):
        outchannels = CONFIG.RESNET.BOTTOM_OUT_PLANES
        ciz = CONFIG.SIMULATED.DATA.CROPPED_IMAGE_SIZE
        if len(ciz) == 3:
            inchannels = ciz[0]
            self.cropped_image_size = ciz[-2:]
        else:
            inchannels = 1
            self.cropped_image_size = ciz

        self.bottom = resnet.ResNetBottom(inchannels, outchannels)
        self.outchannels = self.bottom.outchannels

    def _setup_middle(self, model):
        # block, nlayers, inchannels, stride, multiplier=2
        down = CONFIG.RESNET[model]['down']
        up = CONFIG.RESNET[model]['up']
        inchannels = self.bottom.outchannels
        self.outchannels, self.middle = resnet.build_resnet_middle(
            inchannels, down, up)

    def _setup_top(self, imshape):
        self.top = RPN(self.outchannels, imshape, self.training_flag)
        self.apply_mask = self.top.apply_mask

    @property
    def training_flag(self):
        return self._training_flag

    @training_flag.setter
    def training_flag(self, value):
        self._training_flag = value
        self.training = value & ModelTask.TRAINING

    def forward(self, im):
        # print('imshape: {}'.format(im.size()))
        im = self.bottom(im)
        # print('imshape: {}'.format(im.size()))
        im = self.middle(im)
        # print('imshape: {}'.format(im.size()))
        im = self.top(im)

        return im
