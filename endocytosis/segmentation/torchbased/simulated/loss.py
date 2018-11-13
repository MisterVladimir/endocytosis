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
    """
    Parameters
    -----------
    args : 2-tuples
        arg[0] : name of the tensor to print
        arg[1] : torch.Tensor
    """
    for name, arg in args:
        print("{} size: {}".format(name, arg.size()))


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ratio = CONFIG.SIMULATED.LOSS_RATIO

    def _get_weights(self, mask):
        # total number of pixels in mask
        n = torch.prod(torch.tensor(mask.size())).item()
        foreground = mask.sum().item()
        background = n - foreground
        weights = torch.zeros_like(mask, dtype=torch.float32)
        weights[mask] = 1. / foreground
        weights[~mask] = 1. / background
        return weights

    def forward(self, p, xdeltas, mask, dx, dy):
        """
        Parameters
        -------------
        xdeltas : torch.Tensor (dtype=torch.float32)
        xmask : torch.Tensor (dtype=torch.uint8)
        Trained data output from 

        mask, dx, dy : torch.Tensor
        """
        # ratio of background/foreground pixels
        bg_fg_weights = self._get_weights(mask[0])
        # bg_fg_weights = bg_fg_weights.to(p.device)
        mask2d = mask[0].to(torch.float32)
        mask_loss = nn.functional.binary_cross_entropy(
            p, mask2d, weight=bg_fg_weights)

        deltas = torch.cat([dx, dy], dim=1)
        deltas = torch.masked_select(deltas, mask).reshape((2, -1))
        # print("deltas : {}".format(deltas.size()))
        xdeltas = torch.masked_select(xdeltas, mask).reshape((2, -1))
        # print("xdeltas : {}".format(xdeltas.size()))
        sqdif = (deltas - xdeltas)**2
        deltas_loss = sqdif.sum() / xdeltas.size()[1]
        return 1000. * (mask_loss + self.ratio * deltas_loss) / (1 + self.ratio)
