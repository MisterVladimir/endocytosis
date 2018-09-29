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
from addict import Dict
from scipy import sparse
import h5py
import os
import string
from fiji_tools.io import IO

from endocytosis.io.path import PathFinder

"""
Notes to self
Despite all ROI having same square size, we'll probably still need to
train height and width of each anchor's bounding box to account for movement
of spots from one 'channel' (timepoint) to the next.

Instead of classifying into simply 'background' and 'foreground', we might
classify anchors as 'background', 'start', 'middle', and 'end', the latter
three corresponding to their () during the endocytosis event. Each of these
should have very characteristic appearances although it's unclear whether
the Endocytosis dataset actually identifies the faint GFP signal at the
very early and late stages. I hope this will ultimately help link ROI
over the timecourse.
"""

# old
class EndocytosisDataset(Dataset, IO):
    """
    For loading the Endocytosis dataset.
    ********************************************************************
    ********************************************************************

    Parameters
    -----------
    source: hdf5 File or Group
    hyp: addict.Dict
    Hyperparameters. 
    """
    def __init__(self, source, hyp):
        super().__init__()
        self.source = source
        # hyperparameters
        # TODO: create config file, load its data as globals
        # TODO: alternatively set hyperparameters in a text file, and use pyparsing to
        # input hyperparameters
        self.info = Dict()
        self.info.hyp.ROI_SIDE_LENGTH = 11
        self.info.hyp.CROPPED_IMAGE_SIZE = 64
        self.info.hyp.RANDOM_CROP = True
        if self.info.hyp.RANDOM_CROP:
            self._random_crop_ind = {}
        self.info.hyp.ROI_ATTRIBUTE_NAME = 'centroid'
        self._data_setup()

    def _data_setup(self):
        # set image data
        self.image = self.source['image']
        self.info.IMAGE_SHAPE = np.array(self.image.shape, dtype=int)

        # shorten commonly used variables
        side_length = self.info.hyp.ROI_SIDE_LENGTH
        attr_name = self.info.hyp.ROI_ATTRIBUTE_NAME
        imshape = self.info.IMAGE_SHAPE

        # ground truth data labeling each pixel that contains an object
        gt = np.zeros(self.info.IM_SHAPE, bool)

        # fill ground truth data
        # first put ROI data into sparse array
        grp = self.source[attr_name]
        # number rows = number of ROI; columns are time dimension
        nrow = len(grp.keys())
        ncol = imshape[0]
        assert (isinstance(side_length, (int, np.integer)),
                "side length not integer")

        arr = np.array([(i, int(t), *grp[n][t])
                        for i, n in enumerate(grp.keys())
                        for t in grp[n]]).T
        # roi centroids
        data = arr[2:]
        # row (roi) and column (time) indices
        row = arr[0]
        col = arr[1]
        coo = [sparse.coo_matrix((d, (row, col)), shape=(nrow, ncol))
               for d in data]
        # spx = sparse x; spy = sparse y
        # contains x and y coordinates of roi centroids
        spx = coo[0].tocsr()
        spy = coo[1].tocsr()
        for i in range(nrow):
            x = spx[i]
            y = spy[i]
            ind = x.indices

            x0 = np.maximum(x.data - side_length / 2, 0, dtype=int)
            x1 = np.minimum(x.data + side_length / 2, imshape[0], dtype=int)
            slx = [slice(i, j) for i, j in zip(x0, x1)]

            y0 = np.maximum(y.data - side_length / 2, 0, dtype=int)
            y1 = np.minimum(y.data + side_length / 2, imshape[1], dtype=int)
            sly = [slice(i, j) for i, j in zip(y0, y1)]

            gt[ind, slx, sly] = True

        filename = ''.join(np.random.choice(string.ascii_letters, 15)) + '.h5'
        self._temp_filename = filename
        self._temp_file = h5py.File(filename)
        self.gt = self._temp_file.create_dataset('gt', shape=gt.shape,
                                                 dtype=bool, data=gt)

    def _base_crop(self, t, x, y):
        im = self.image[t, x, y]
        gt = self.gt[t, x, y]
        # put into torch order (C, H, W)
        return (np.swapaxes(im, 1, 2), np.swapaxes(gt, 1, 2))

    def _new_random_crop(self):
        sidelength = self.info.hyp.CROPPED_IMAGE_SIZE
        cropsize = np.full(2, sidelength, dtype=int)
        max_start = self.info.IM_SHAPE[1:] - cropsize
        x0, y0 = [np.random.randint(0, i) for i in max_start]
        x1, y1 = [x0, y0] + cropsize
        t0 = np.random.randint(0, self.info.IM_SHAPE[0] - 2)
        return (slice(t0, t0+3), slice(x0, x1), slice(y0, y1))

    def _make_orderly_crop(self, index):
        imshape = self.info.IM_SHAPE
        sidelength = self.info.hyp.CROPPED_IMAGE_SIZE
        cropsize = np.ones(3, dtype=int)*[1, sidelength, sidelength]
        index_shape = imshape // cropsize
        start = (imshape % cropsize) // 2
        unraveld_index = np.unravel_index([index], index_shape)
        t0, x0, y0 = start + unraveld_index*cropsize
        t1, x1, y1 = start + (unraveld_index + (3, 1, 1))*cropsize
        return (slice(t0, t1), slice(x0, x1), slice(y0, y1))

    def __getitem__(self, key):
        if self.info.hyp.RANDOM_CROP:
            # XXX: should we save self._random_crop_ind to self.info.hyp?
            try:
                t, x, y = self._random_crop_ind[key]
            except KeyError:
                t, x, y = self._new_random_crop()
                self._random_crop_ind[len(self._random_crop_ind)] = (t, x, y)
        else:
            t, x, y = self._make_orderly_crop(key)

        im, gt = self._base_crop(t, x, y)
        return (from_numpy(im), from_numpy(gt))

    def __len__(self):
        sh = self.info.IM_SHAPE - (2, 0, 0)
        cropsize = self.info.CROPPED_IMAGE_SIZE
        sh[1:] = sh[1:] // cropsize
        return np.prod(sh)

    def cleanup(self):
        try:
            self._temp_file.close()
            os.remove(self._temp_filename)
        except AttributeError:
            pass
