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
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import h5py
import numpy as np

import endocytosis.contrib.PYME.localization.ofind as _ofind


class ObjectIdentifier(_ofind.ObjectIdentifier):
    def _get_cropped_roi(self, x, y, size):
        n = len(self.x)
        # xy coördinates as n by 2 array
        xy = np.array([self.x[:], self.y[:]], int).T
        # remove any coördinates close to the edge
        mask = np.all(size / 2 < xy < self.data.shape[-2:] - size / 2,
                      axis=1)
        xy = xy[:, mask][:, :, None]
        # start and stop indices of crop
        bounds = xy + np.array([-size // 2, size // 2], int)[None, None, :]
        data = np.empty((n, *self.data.shape), dtype=np.float32)
        data[:] = [self.data[:, slice(*x), slice(*y)] for x, y in bounds]
        return data

    def write(self, path, roi_side_length):
        data = self._get_cropped_roi(self.x, self.y, roi_side_length)
        with h5py.File(path) as f:
            _ = f.create_dataset('image', data.shape, float, data)
