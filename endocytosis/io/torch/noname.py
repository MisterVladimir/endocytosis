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
from addict import Dict

from endocytosis.io import IO
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


class EndocytosisDataset(Dataset, IO):
    """
    For loading the Endocytosis dataset.
    ********************************************************************
    Note that if shuffling is turned on, ROI served up by the
    __getitem__ method of this class lose their time-dependent
    ordering. This class is simply for identifying the 'objectness'
    of a region like in Faster R-CNN. It may be a future consideration
    to have each ROI/image pair represented by a single
    torch.utils.data.Dataset and then link such Datasets with a
    ConcatDataset.
    ********************************************************************

    Parameters
    -----------
    im_reader: endocytosis.io.image.ImageReader
    ImageReader already loaded with a dataset.

    roi_dir: endocytosis.io.roi.IJZipReader
    IJZipReader already loaded with ImageJ ROI.
    """
    def __init__(self, im_reader, roi_reader):
        super().__init__()
        self.im_reader = im_reader
        self.roi_reader = roi_reader
        # 
        self.imshape = 
        # hyperparameters
        # TODO: create config file, load these globals
        # TODO: alternatively set these in a text file, and use pyparsing to
        self.hyp = Dict()
        # input hyperparameters
        self.hyp.ROI_SQUARE_SIZE = 11
        self.hyp.CROPPED_IMAGE_SIZE = 64
        self.hyp.RANDOM_CROP = True

    def __getitem__(self, key):
        # index is roi's index, not necessarily the timepoint
        filename, index = key
        roi = self.roi_reader[filename][index:index+2]
        tstart = roi[0].t
        centroids = np.array([r.centroid['px'] for r in roi])

        # 
        if self.hyp.RANDOM_CROP:
            x, y = self._random_crop(), self._random_crop()
        else:
            x, y = slice(None), slice(None)
        im = self.im_reader.data.request(0, 0, slice(tstart, tstart+2), x, y)


    def __len__(self):
        pass

    def _random_crop_slice(self, max):
        """
        max: int
        Maximum dimension size.
        """

    def cleanup(self):
        try:
            pass
        except AttributeError:
            pass
