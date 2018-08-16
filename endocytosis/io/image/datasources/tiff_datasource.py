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

import endocytosis.contrib.gohlke.tifffile as tifffile
from endocytosis.io.image.datasources.base_datasource import (BaseImageRequest,
                                                              BaseDataSource)
from endocytosis.io.path import PathFinder


class TiffPathFinder(PathFinder):
    def __init__(self, regexp=r'.*'):
        super().__init__(regexp, 'tif')


class TiffImageRequest(BaseImageRequest):
    """
    Used to request images from a Tiff DataSource. Call with the
    C, T, Z, X, Y dimensions of the slice of the image we want.

    order: str
        Concatenated string of image dimension order.
    shape : iterable of length 5
        Shape of each dimension in order CTZXY. Having a known order is helpful
        because the (ome) tiff metadata only states the number of
        channels, z slices, and timepoints. To get the true 'shape', one must
        factor in the dimension order. This is done in the 'shape' property's
        setter.

    Example
    ---------
#    a 3-color image with 8 z slices
    req = TiffImageRequest('TCZXY', 1, 3, 8) # must be c, t, z order
    datasource = TiffDataSource(open_file, req)
#    retrive 3d image in the first channel, third timepoint, and all z slices
#    call the datasource's request method with arguments in C, T, Z order
    z_stack = datasource.request(0,2,slice(None))
    """
    module_name = 'tiff_datasource'

    def __call__(self, *ctzxy):
        if len(ctzxy) == 5:
            self['CTZXY'] = ctzxy
        elif len(ctzxy) == 3:
            ctzxy = list(ctzxy) + [slice(None), slice(None)]
            self['CTZXY'] = ctzxy
        elif len(ctzxy) == 0:
            # use previous contents of self
            pass
        else:
            raise Exception('')

        # get order of the C, T, and Z dimensions
        order = self.dimension_order.replace('X', '').replace('Y', '')
        arr = np.array(self[order])[:, None]
        imshape = [self.image_shape[k] for k in order]
        # locate their raveled index
        return (np.ravel_multi_index(arr, imshape, order='F')[0],
                self['X'], self['Y'])


class TiffDataSource(BaseDataSource):
    module_name = 'tiff_datasource'

    def __init__(self, path, request):
        super().__init__()
        self._request = request
        self.datasource = tifffile.TiffFile(path)

    def request(self, *ctzxy):
        """
        Arguments
        -----------
        request : ImageDataRequest or list
        Each item in the list is a slice or int corresponding to the
        (channel, time, axial position) in that order.
        """
        n, x, y = self._request(*ctzxy)
        return self.datasource.pages[n].asarray()[x, y]

    def cleanup(self):
        self.reader.close()
