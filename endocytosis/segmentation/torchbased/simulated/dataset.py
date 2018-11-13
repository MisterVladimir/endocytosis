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
from torch.utils.data import (Dataset, ConcatDataset, DataLoader)
from torch import from_numpy
import h5py
import numbers
from vladutils.io import IO

from ....config import CONFIG
from ..enums import ModelTask


class SimulationDataset(Dataset, IO):
    """
    Parameters
    -----------
    path : str
        hdf5 file name with full path.

    training : ModelTask
        Enum options are TRAINING, TESTING, INFERENCE
    """
    def __init__(self, path, training):
        super().__init__()
        self.training_flag = training
        self.h5file = h5py.File(path, 'r')
        self._parameter_setup()
        self._data_setup()

    @property
    def training_flag(self):
        return self._training_flag

    @training_flag.setter
    def training_flag(self, value):
        self._training_flag = value
        self.training = value & ModelTask.TRAINING

    @property
    def cropped_image_size(self):
        return self._cropped_image_size

    @cropped_image_size.setter
    def cropped_image_size(self, ciz):
        """
        Sets the shape of the image returned by self.__getitem__

        Parameters
        ------------
        ciz : int or list of ints
            If int, output image shape is [1, ciz, ciz].
            If list of length two, we assume it's a grayscale image; 
            otherwise if ciz contains three integers, shape is same
            as the argument.
        """
        if isinstance(ciz, numbers.Integral):
            self._cropped_image_size = np.array([1, ciz, ciz])
        elif len(ciz) == 2:
            self._cropped_image_size = np.array([1, *ciz])
        elif len(ciz) == 3:
            self._cropped_image_size = np.array(ciz)
        else:
            raise TypeError('cropped_image_size must be set with an integer '
                            'or a list of length two or three.')

    def _parameter_setup(self):
        self.roi_attribute_name = CONFIG.SIMULATED.DATA.ROI_ATTRIBUTE_NAME
        self.cropped_image_size = CONFIG.SIMULATED.DATA.CROPPED_IMAGE_SIZE
        self.random_crop = CONFIG.SIMULATED.TRAIN.RANDOM_CROP
        self.normalize_image_data = CONFIG.SIMULATED.DATA.NORMALIZE_IMAGE_DATA
        if self.normalize_image_data:
            self.image_data_mean = CONFIG.SIMULATED.DATA.IMAGE_DATA_MEAN
            self.image_data_stdev = CONFIG.SIMULATED.DATA.IMAGE_DATA_STDEV

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
        For use with output of _make_orderly_crop or _make_random_crop
        methods.

        Parameters
        ------------
        t, x, y: slice

        Returns
        ------------
        im: torch.Tensor
        Cropped image data.

        mask: torch.Tensor
        Boolean array, denotes whether pixel contains a spot centroid.

        deltas: torch.Tensor
        XY coördinates of spot centroid relative to the top left corner (0, 0)
        of the pixel.
        """
        im = self.image[t, x, y].astype(np.float32)
        if self.normalize_image_data:
            mean = self.mean[t]
            stdev = self.stdev[t] / self.image_data_stdev
            im = (im - mean) / stdev + self.image_data_mean

        if self.training_flag & ModelTask.INFERENCE:
            out = [im, ] + [np.array([i.start]) for i in (t, x, y)]
            return [from_numpy(item) for item in out]

        # For each pixel that contains a spot centroid, assign it that
        # centroid's (x, y) coördinates. Note that this coördinate is
        # relative to the top left corner of the cropped image, not the
        # original image.
        # XXX: this only works for grayscale images
        deltas = np.zeros([2, *im.squeeze().shape], dtype=np.float32)
        # XY coördinates from ground truth data
        gt = self.h5file['ground_truth'][self.roi_attribute_name][str(t.start)]
        gt = gt.value
        # filter out any coördinates outside this cropped image
        # cmask stands for Coördinate Mask
        cmask = np.logical_and(gt >= [x.start, y.start], gt < [x.stop, y.stop])
        cmask = cmask.all(1)
        if np.any(cmask):
            xy = gt[cmask] - [x.start, y.start]
            xy = xy.T
            xi, yi = xy.astype(int)
            # (0, 0) coördinate is the center of the upper left hand
            # corner's pixel
            deltas[:, xi, yi] = xy
            # pixels where a spot centroid is located
            mask = deltas > 0
            mask = mask.all(0)[None, :]
            # TODO: check whether mask should be float32 or long
            out = (im, mask.astype(np.uint8))
        else:
            out = (im, np.zeros_like(im, dtype=np.uint8))

        dx, dy = deltas[:, None, :, :]
        out += (dx, dy)
        return [from_numpy(item) for item in out]

    def _new_random_crop(self):
        max_start = self.imshape - self.cropped_image_size
        t0, x0, y0 = [np.random.randint(0, i) for i in max_start]
        t1, x1, y1 = t0, x0, y0 + self.cropped_image_size
        return (slice(t0, t1), slice(x0, x1), slice(y0, y1))

    def _make_orderly_crop(self, index):
        """
        It's easiest to explain this by example

        If the original data is 2d and of shape
        _________________
        |               |
        |               |
        |               |
        |_______________|

        and we want to return cropped images of shape
        _________
        |       |
        |_______|

        This method returns the slices corresponding to the following:
        _________________
        |  i=0  |  i=1  |
        |_______|_______|
        |  i=2  |  i=3  |
        |_______|_______|

        where i is the index.

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

        output = self._base_crop(*txy)

        # if self.training_flag & ModelTask.TRAINING:
        #     output[0].requires_grad_()

        return output

    def __len__(self):
        return np.prod(self.imshape // self.cropped_image_size)

    def get_txy(self, index):
        if self.random_crop:
            t, x, y = self._random_crop_ind[index]
        else:
            t, x, y = self._make_orderly_crop(index)

        return t.start, x.start, y.start

    def cleanup(self):
        try:
            self.h5file.close()
        except AttributeError:
            pass
