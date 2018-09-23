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
import os
import copy

import endocytosis.contrib.PYME.localization.ofind as _ofind


class ObjectIdentifier(_ofind.ObjectIdentifier):
    def __init__(self, data, filterMode="fast", filterRadiusLowpass=1,
                 filterRadiusHighpass=3, filterRadiusZ=4):
        if isinstance(data, h5py.Dataset):
            self.filename = data.file.filename
            self.name = data.name
            data = data.value
        else:
            self.filename = None
            self.name = None

        super().__init__(data, filterMode="fast", filterRadiusLowpass=1,
                         filterRadiusHighpass=3, filterRadiusZ=4)

    def _get_cropped_roi(self, x, y, size):
        n = len(x)
        # xy coördinates as n by 2 array
        xy = np.array([x, y], int).T
        # remove any coördinates close to the edge
        mask = np.all(size / 2 < xy < self.data.shape[-2:] - size / 2,
                      axis=1)
        xy = xy[:, mask][:, :, None]
        # start and stop indices of crop
        bounds = xy + np.array([-size // 2, size // 2], int)[None, None, :]
        data = np.empty((n, *self.data.shape), dtype=np.float32)
        data[:] = [self.data[:, slice(*x), slice(*y)] for x, y in bounds]
        return data

    def write(self, roi_side_length, folder='', filename=''):
        """
        Given object centroids identified by the FindObjects method,
        save the bounding boxes of roi_side_length as an HDF5 Dataset.

        roi_side_length: int

        folder: str
        Location of HDF5 File object. Can be an absolute path or path
        relative to the current working directory.

        filename, name: str
        Name of the HDF5 file and dataset name to save the data as. Note
        that if this object's constructor's 'data' argument was an
        h5py.Dataset, and filename and name are left as empty strings,
        we save the data to that file and the directory the Dataset
        is in. Otherwise, if filename is an empty string, file is saved
        as 'ofind.hdf5' in the current working directory. The saved Dataset
        is always named "spots".
        """
        self.folder = os.path.abspath(folder)
        if filename:
            self.filename = filename
        elif self.filename:
            filename = self.filename
        else:
            filename = 'ofind.hdf5'
        path = os.path.join(self.folder, filename)

        if self.name:
            name = '/'.join(self.name.split('/')[:-1])
        else:
            name = ''
        name += '/spots'

        data = self._get_cropped_roi(self.x[:], self.y[:], roi_side_length)

        with h5py.File(path) as f:
            _ = f.create_dataset(name, data.shape, data=data)
