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

from endocytosis.io.image.datasources.base_datasource import FileDataSource
from endocytosis.helpers.error import DatasourceError


class ArrayDatasource(FileDataSource):
    """
    Wrapper around numpy.ndarray image data such that we can use existing
    NestedDatasource objects to filter the final output data.

    Arguments
    -----------
    datasource: numpy.ndarray
    Image data, dimensions are channel, time, z-slice, x, y. If fewer than
    5 dimensions are supplied, add dimensions.
    """
    module_name = 'array_datasource'

    def __init__(self, data, *args):
        super().__init__()
        dim_expansion = (None, )*(5 - data.ndim)
        self.datasource = data[dim_expansion]

    def request(self, c=0, t=0, z=0, x=slice(None), y=slice(None)):
        im = self.datasource[c, t, z, x, y]
        if im.ndim > 3:
            raise DatasourceError('Output must be 3d.')
        elif im.ndim == 3:
            pass
        elif im.ndim == 2:
            # output should be shaped (1, *, *)
            im = np.moveaxis(np.atleast_3d(im), 2, 0)
        return im

    def cleanup(self):
        pass
