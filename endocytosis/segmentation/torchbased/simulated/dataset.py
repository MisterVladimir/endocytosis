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
import numpy as np
import torch
from torch.utils.data import (Dataset, ConcatDataset, DataLoader)
from torch import from_numpy
from addict import Dict
from scipy import sparse
import h5py
import os
import string
import numbers

from ....io import IO
from ....config import CONFIG


class SimulatedDataset(Dataset, IO):
    """
    Parameters
    -----------
    path : str
        hdf5 file name with full path.

    training : bool
        True if training, False if making inferences.
    """
    def __init__(self, path, training):
        super().__init__()
        self.training = training
        self.h5file = h5py.File(path, 'r')
        self._parameter_setup()
        self._data_setup()

    @property
    def cropped_image_size(self):
        return self._cropped_image_size

    @cropped_image_size.setter
    def cropped_image_size(self, ciz):
        if isinstance(ciz, numbers.Integral):
            self._cropped_image_size = np.array([1, ciz, ciz])
        elif len(ciz) == 2:
            self._cropped_image_size = np.array([1, *ciz])
        elif len(ciz) == 3:
            self._cropped_image_size = np.array(ciz)
        else:
            raise TypeError('cropped_image_size must be set with list of '
                            'length two or three.')

    def _parameter_setup(self):
        self.roi_attribute_name = CONFIG.TRAIN.SIMULATED.ROI_ATTRIBUTE_NAME
        self.cropped_image_size = CONFIG.TRAIN.SIMULATED.CROPPED_IMAGE_SIZE
        self.random_crop = CONFIG.TRAIN.SIMULATED.RANDOM_CROP
        self.normalize_image_data = CONFIG.TRAIN.SIMULATED.NORMALIZE_IMAGE_DATA
        if self.normalize_image_data:
            self.image_data_mean = CONFIG.TRAIN.SIMULATED.IMAGE_DATA_MEAN
            self.image_data_stdev = CONFIG.TRAIN.SIMULATED.IMAGE_DATA_STDEV

    def _data_setup(self):
        # set image data
        self.image = self.h5file['image']['data']
        self.mean = self.h5file['image']['mean']
        self.stdev = self.h5file['image']['stdev']
        self.imshape = np.array(self.image.shape, dtype=np.uint16)

        if self.random_crop:
            self._random_crop_ind = {}

    def _base_crop(self, t, x, y):
        """
        Parameters
        ------------
        t, x, y: slice

        Returns
        ------------
        im: numpy.ndarray
        Cropped image data.

        mask: numpy.ndarray
        Boolean array, denotes whether pixel contains a spot.

        deltas: numpy.ndarray
        XY coördinates of spot centroid relative to the top left corner (0, 0)
        of the pixel.
        """
        im = self.image[t, x, y]
        if self.normalize_image_data:
            mean = self.mean[t]
            stdev = self.stdev[t] / self.image_data_stdev
            im = (im - mean) / stdev + self.image_data_mean

        if not self.training:
            return from_numpy(im), None, None

        deltas = np.zeros([2, *im.shape[1:]], dtype=np.float32)
        # XY cooridnates from ground truth data
        gt = self.h5file['ground_truth'][self.roi_attribute_name][str(t.start)]
        gt = gt.value
        # filter out any coördinates outside this cropped image
        mask = np.logical_and(gt >= [x.start, y.start], gt < [x.stop, y.stop])
        mask = mask.all(1)
        if np.any(mask):
            # x coordinates, y coordinates within the cropped image
            xy = gt[mask] - [x.start, y.start]
            xyint = xy.astype(int)
            xi, yi = xyint.T
            deltas[:, xi, yi] = (xy - xyint).T
            mask = deltas[0] > 0
            out = (im, mask.astype(np.float32), deltas)
        else:
            out = im, np.zeros_like(im, dtype=np.float32), deltas
        return (from_numpy(item) for item in out)

    def _new_random_crop(self):
        max_start = self.imshape - self.cropped_image_size
        t0, x0, y0 = [np.random.randint(0, i) for i in max_start]
        t1, x1, y1 = t0, x0, y0 + self.cropped_image_size
        return (slice(t0, t1), slice(x0, x1), slice(y0, y1))

    def _make_orderly_crop(self, index):
        """
        Returns
        ----------
        t, x, y: slice
        """
        index_shape = self.imshape // self.cropped_image_size
        unraveled_index = np.unravel_index(index, index_shape)
        unraveled_index = np.array(unraveled_index)
        # in case cropped_image_size doesn't divide evenly into imshape
        start = (self.imshape % self.cropped_image_size) // 2
        t0, x0, y0 = start + unraveled_index*self.cropped_image_size
        t1, x1, y1 = start + (unraveled_index + 1)*self.cropped_image_size
        return (slice(t0, t1), slice(x0, x1), slice(y0, y1))

    def __getitem__(self, key):
        if self.random_crop:
            try:
                txy = self._random_crop_ind[key]
            except KeyError:
                txy = self._new_random_crop()
                i = len(self._random_crop_ind)
                self._random_crop_ind[i] = txy
        else:
            txy = self._make_orderly_crop(key)

        im, mask, deltas = self._base_crop(*txy)

        keys = ('im', 'mask', 'deltas')
        if self.training:
            im.has_grad = True
        else:
            keys = keys + ('t', 'x', 'y')

        values = (im, mask, deltas) + tuple((i.start for i in txy))
        return dict(zip(keys, values))

    def __len__(self):
        return np.prod(self.imshape // self.cropped_image_size)

    def cleanup(self):
        try:
            self.h5file.close()
        except AttributeError:
            pass
