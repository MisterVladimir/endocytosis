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
from collections import MutableSequence, Callable

from endocytosis.io import IO


class BaseImageRequest(MutableSequence, Callable):
    """
    Used to request images from a DataSource. Call with the
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
    datasource = TiffFileDataSource(open_file, req)
#    retrive 3d image in the first channel, third timepoint, and all z slices
#    call the datasource's request method with arguments in C, T, Z order
    z_stack = datasource.request(0,2,slice(None))
    """
    module_name = 'base_datasource'

    def __init__(self, order, *shape):
        """
        order : str
        Listed in same order as mdh['DimensionOrder'], e.g. TZCYX

        shape : iterable
        Shape of data in CTZXY order.
        """
        # TODO: make sure shape is 3 units long; otherwise, add 1s
        self.order = order.upper()
        self.shape = shape
        self._positions = {'C': 0, 'T': 1, 'X': 3, 'Y': 4, 'Z': 2}
        self._entries = [0, 0, 0, slice(None), slice(None)]

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, s):
        # coerce to c, t, z, x, y order
        c, t, z, x, y = s
        d = {'C': c, 'T': t, 'Z': z, 'X': x, 'Y': y}
        self._shape_dict = d
        self._shape = [d[key] for key in self.order]

    def __repr__(self):
        d = {'order': self.order, **self._shape_dict}
        return 'ImageRequest(order: {order}; CTZ: ({C}, {T}, {Z}))'.format(**d)

    def __setitem__(self, key, value):
        if isinstance(value, int):
            value = slice(slice(value, value+1, None).indices(
                                                    self._shape[key]))
        if isinstance(key, int):
            # example : self[0] = slice(2,4)
            self._entries[key] = value
        elif isinstance(key, str):
            if len(key) == 1:
                # example : self['C'] = 2
                i = self._positions[key]
                # print (key, i)
                self._entries[i] = value
            elif len(key) > 1:
                # example : self['CZ'] = (2, slice(1))
                for i, char in enumerate(key):
                    self[char] = value[i]
        else:
            raise TypeError('Incompatible key or value for __setitem__.')

    def __getitem__(self, key):
        # print (key)
        if isinstance(key, int):
            return self._entries[key]
        elif isinstance(key, str):
            if len(key) == 1:
                i = self._positions[key]
                return self._entries[i]
            elif len(key) > 1:
                return [self[k] for k in key]
        else:
            raise TypeError('Incompatible key or value for __getitem__.')

    def __delitem__(self, key):
        raise NotImplementedError

    def __len__(self):
        return 5

    def __contains__(self, index):
        n = len(index)
        return np.all(np.asarray(self.shape[:n]) - np.asarray(index) > 0)

    def insert(self, index, value):
        raise NotImplementedError('')

    def __call__(self, key=None, value=None):
        if key is not None and value is not None:
            self[key] = value
        return {'C': self[0], 'T': self[1], 'Z': self[2]}


class BaseDataSource(IO):
    module_name = 'base_datasource'
