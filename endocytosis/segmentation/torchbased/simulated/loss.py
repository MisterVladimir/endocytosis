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

from ....config import CONFIG


def print_sizes(*args):
    for name, arg in args:
        print("{} size: {}".format(name, arg.size()))


class MaskLoss(nn.Module):
    def __init__(self, inchannels):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduction='elementwise_mean')
        self.conv = nn.Conv2d(inchannels, out_channels=2, kernel_size=1)

    def forward(self, x, mask):
        x = self.conv(x)
        mask = mask.to(dtype=torch.long).squeeze()
        return self.loss(x, mask)


class DeltasLoss(nn.Module):
    def __init__(self, inchannels):
        super().__init__()
        # self.loss = nn.SmoothL1Loss(reduction='elementwise_mean')
        self.conv = nn.Conv2d(inchannels, out_channels=2, kernel_size=1)

    def forward(self, x, mask, dx, dy):
        mask = mask.to(torch.uint8)
        x = self.conv(x)
        x = torch.masked_select(x, mask)
        deltas = torch.stack([dx, dy])
        deltas = torch.masked_select(deltas, mask)
        x = (x - deltas)**2
        x = x.reshape((2, -1))
        x = x.sum(0)
        return x.mean()


class CombinedLoss(nn.Module):
    def __init__(self, inchannels):
        super().__init__()
        self.ratio = CONFIG.LOSS_RATIO
        self.deltas_loss = DeltasLoss(inchannels)
        self.mask_loss = MaskLoss(inchannels)

    def forward(self, x, mask, dx, dy):
        mask_loss = self.mask_loss(x, mask)
        deltas_loss = self.deltas_loss(x, mask, dx, dy)
        return mask_loss + self.ratio * deltas_loss
