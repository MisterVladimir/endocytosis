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
from collections import OrderedDict

from endocytosis.io import IO


class IntIndexDict(OrderedDict):
    def __getitem__(self, key):
        if isinstance(key, int):
            key = list(self.keys())[key]
            return super().__getitem__(key)
        elif isinstance(key, slice):
            key = list(self.keys())[key]
            return [OrderedDict.__getitem__(self, k) for k in key]
        else:
            return super().__getitem__(key)

    def __setitem__(self, key, value):
        if isinstance(key, int):
            key = list(self.keys())[key]
            return super().__setitem__(key, value)
        elif isinstance(key, slice):
            ind = range(len(self))[key]
            for i in ind:
                super().__setitem__(key[i], value[i])
        else:
            super().__setitem__(key, value)


class BaseImageRequest(IntIndexDict):
    """
    Used to request images from a Tiff DataSource. Call with the
    C, T, Z, X, Y dimensions of the slice of the image we want.

    Currently only 3-dimensional outputs are supported. That is,
    only one of the arguments to self.__call__() may be a slice
    object.

    order: str
        Concatenated string of image dimension order.
    shape : iterable of length 5
        Shape of each dimension in order CTZXY. 

    Example
    ---------
#    a 3-color image with 8 z slices, each slice 512x512
    req = TiffImageRequest('TCZXY', 3, 1, 8, 512, 512)
    datasource = TiffDataSource(image_path, req)

#    retrive image in the first channel, third timepoint, and all z slices
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
        super().__init__()
        # TODO: make sure shape is 5 units long; otherwise, add 1s
        self.ctzxy_order = order.upper()
        self.ctz_order = self.ctzxy_order.replace('X', '').replace('Y', '')
        # shape of image data, in image's true dimension (channel, time, z) order
        self.image_shape = IntIndexDict(zip(self.ctzxy_order,
                                            [None]*len(order)))
        self.image_shape.update(dict(zip('CTZXY', shape)))
        self.__setitem__('CTZXY', [0, 0, 0, slice(None), slice(None)])

    def __setitem__(self, key, value):
        if isinstance(key, str) and len(key) > 1:
            for i, k in enumerate(key):
                self.__setitem__(k, value[i])
        else:
            super().__setitem__(key, value)

    def __getitem__(self, key):
        if isinstance(key, str) and len(key) > 1:
            return [self.__getitem__(k) for k in key]
        else:
            return super().__getitem__(key)

    def __delitem__(self, key):
        raise NotImplementedError

    def __len__(self):
        return 5

    def insert(self, index, value):
        raise NotImplementedError('')

    def __call__(self, *ctzxy):
        raise NotImplementedError('')


class BaseDataSource(IO):
    module_name = 'base_datasource'
