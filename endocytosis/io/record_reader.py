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

import tensorflow as tf
import glob

from endocytosis.io import IO

features = {'image': tf.FixedLenFeature([], tf.string),
            'centers': tf.VarLenFeature(tf.float32),
            'n': tf.FixedLenFeature([], tf.int64)
            }


class RecordReader(IO):
    """
    Passes data from a tensorflow record file to the neural network graph.
    """
    def __init__(self, path=None, data=None, mdh=None):
        super().__init__()
        self._repath = path
        self._data = data
        self._mdh = mdh

        if path is not None:
            self.load_from_path(path)
        elif data is not None and mdh is not None:
            self.load_from_data(data, mdh)

    def load_from_path(self, path):
        pass

    def load_from_data(self, data):
        pass

    def write(self):
        pass
