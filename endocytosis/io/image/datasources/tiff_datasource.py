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
import copy

import endocytosis.contrib.gohlke.tifffile as tifffile
from endocytosis.io.image.datasources.base_datasource import (BaseImageRequest,
                                                              BaseDataSource)
from endocytosis.io.path import PathFinder


class TiffPathFinder(PathFinder):
    def __init__(self, regexp=r'.*'):
        super().__init__(regexp, 'tif')


class TiffImageRequest(BaseImageRequest):
    __doc__ = BaseImageRequest.__doc__
    module_name = 'tiff_datasource'

    def _get_page_indices(self):
        order = self.ctz_order
        # shape of the tif image, in its DimensionOrder
        tif_shape = [self.image_shape[k] for k in order]

        # the index selfuested, stored as values in self, is rearranged to the
        # dimension order that the tiff image is in
        index = np.array(self[order])
        from_ints = np.array([0 if isinstance(i, slice) else i
                              for i in index])[:, None]

        # check if any slice objects have been passed in
        # convert any slices to lists of integers
        is_slice = [isinstance(item, slice) for item in index]
        n_slice_objects = sum(is_slice)
        if n_slice_objects > 1:
            raise TypeError('Only one slice item may be present in the '
                            'request. {} were used.'.format(n_slice_objects))
        elif n_slice_objects == 0:
            as_integers = from_ints
        else:
            # convert slice objects to lists of integers
            # where the index is an integer, leave a placeholder in the form
            # of an empty list
            from_slices = [np.arange(*j.indices(i))
                           if isinstance(j, slice)
                           else [] for i, j in zip(tif_shape, index)]
            # replace the placeholders with zeros
            length = max([len(item) for item in from_slices])
            for i in range(len(index)):
                if not is_slice[i]:
                    from_slices[i] = np.zeros(length, int)
            from_slices = np.concatenate(from_slices).reshape((-1, length))
            # replace all the zeros with indices passed in as integers
            as_integers = from_ints + from_slices
        # identify the tif page data is in
        page_indices = np.ravel_multi_index(as_integers,
                                            tif_shape, order='C')
        return page_indices

    def __call__(self, *ctzxy):
        old_indices = copy.copy(self['CTZXY'])
        # cache the index
        if len(ctzxy) == 5:
            self['CTZXY'] = ctzxy
        elif len(ctzxy) == 3:
            ctzxy = list(ctzxy) + [slice(None), slice(None)]
            self['CTZXY'] = ctzxy
        elif len(ctzxy) == 0:
            # use previous contents of self
            pass
        else:
            return None, None, None
        try:
            n = self._get_page_indices()
            return n, self['X'], self['Y']
        except (TypeError, IndexError) as e:
            # roll back to previous request
            if not len(ctzxy) == 0:
                self['CTZXY'] = old_indices
            raise e


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
        return self.datasource.asarray(key=n)[x, y]

    def cleanup(self):
        self.datasource.close()
