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
import numpy as np

from endocytosis.io.image.datasources import NestedDatasource


class TorchDatasource(NestedDatasource):
    """
    """
    def __init__(self, data):
        super().__init__(data)
        self.datasource = data.datasource

    def request(self, *ctzxy):
        im = self.datasource.request(*ctzxy).astype(np.float32)
        if im.ndim == 2:
            im = im[None, :]
        return torch.from_numpy(im)
