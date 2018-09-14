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
import unittest
import numpy as np
import os.path as path
import glob

from endocytosis.io.roi.file_reading import RoiPathFinder, IJZipReader


class test_IJ_zip_reader(unittest.TestCase):
    base_path = (path.abspath(path.dirname(__file__)) +
                 '{}data{}'.format(path.sep, path.sep))

    def test_basic_read(self):
        result = None
        with IJZipReader(self.base_path + 'test_roi_basic.zip', sep='-') as reader:
            reader.read()
            result = reader._data
        # for k, v in result.items():
            # print("key: {}".format(str(k).strip()))
            # print("{}: {}".format(str(k).strip(), v))
            # print("{}".format(v.args))

    def test_hierarchical_read(self):
        result = None
        with IJZipReader(self.base_path + 'test_roi_hierarchical.zip', sep='-') as reader:
            reader.read()
            result = reader._data
        for k, v in result.items():
            print("key: {}".format(k))
            print("{}: {}".format(k, v))
            # print("{}".format(v.args))

if __name__ == '__main__':
    unittest.main()
