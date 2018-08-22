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
import random

from endocytosis.io.image.datasources import NestedDatasource
from endocytosis.helpers.errors import DatasourceError


class BufferedDatasource(NestedDatasource):
    """
    """
    max_buffer_size = 50

    def __init__(self, data):
        super().__init__(data)
        self.datasource = data.datasource
        self._buffer = {}

    def request(self, *ctzxy):
        ctzxy = list(ctzxy)
        ctz, x, y = ctzxy[:3], ctzxy[3], ctzxy[4]
        try:
            return self._buffer[ctz][x, y]
        except KeyError:
            if len(self._buffer) >= self.max_buffer_size:
                key = random.choice(self._buffer.keys())
                del self._buffer[key]
            im = self.datasource.request(*ctz)
            self._buffer[ctz] = im
            return im[x, y]
