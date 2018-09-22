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
import os
import numpy as np

from endocytosis.test import run
import endocytosis.io.image.image_reader as imreader


class ImageReaderTest(unittest.TestCase):
    folder = os.path.join(os.path.dirname(__file__), 'data')

    def test_load_ImageJ(self):
        filename = 'im0.tif'
        path = os.path.join(self.folder, filename)
        with imreader.ImageReader() as ijreader:
            ijreader.load(path)

    def test_index_ImageJ(self):
        """
        Tests whether single tiff pages can be loaded by passing
        integers into imreader.data.request
        """
        filename = 'im0.tif'
        path = os.path.join(self.folder, filename)
        with imreader.ImageReader() as ijreader:
            ijreader.load(path)
            # every image has a first index
            c0 = ijreader.data.request(0, 0, 0)[:, slice(5), slice(6)]
            _c0 = ijreader.data.request(0, 0, 0, slice(5), slice(6))
            self.assertListEqual(list(c0.shape), list(_c0.shape))
            # confirm that we can request the
            # 2nd color, 1st timepoint, 1st z-slice
            c1 = ijreader.data.request(1, 0, 0)
            # if no arguments passed to 'request', use the previous
            # arguments
            _c1 = ijreader.data.request()
            self.assertTrue(np.all(_c1 == c1))

    def test_slice_ImageJ(self):
        """
        Loading multiple tiff pages by passing slices into
        imreader.data.request
        """
        filename = 'im0.tif'
        path = os.path.join(self.folder, filename)
        with imreader.ImageReader() as ijreader:
            ijreader.load(path)
            # every image has a first index
            c = ijreader.data.request(slice(2), 0, 0, slice(10), slice(10))
            c0 = ijreader.data.request(0, 0, 0, slice(10), slice(10))
            c1 = ijreader.data.request(1, 0, 0, slice(10), slice(10))
            _c = np.concatenate([c0, c1], axis=0)
            self.assertTrue(np.all(c == _c))

    def test_read_ImageJ_errors(self):
        filename = 'im0.tif'
        path = os.path.join(self.folder, filename)
        with imreader.ImageReader() as ijreader:
            ijreader.load(path)
            with self.assertRaises(ValueError) as _:
                # throws ValueError when ctz index isn't available
                a = ijreader.data.request(0, 1, 0)
        # should not raise error because the first tiff page is automatically
        # loaded into the buffer
        c0 = ijreader.data.request(0, 0, 0)
        with self.assertRaises(RuntimeError) as _:
            # raises error because the TiffFile is closed and the tiff page
            # at channel=1, t=0, z=0 was never loaded into the buffer
            # had we requested this ctz index, this wouldn't throw an error
            a = ijreader.data.request(1, 0, 0)

    def test_read_ImageJ_metadata(self):
        pass

    def test_read_OME(self):
        pass

    def test_read_OME_metadata(self):
        pass

# add this to all test modules
TESTS = [ImageReaderTest, ]


def run_module_tests():
    run(TESTS)

if __name__ == '__main__':
    run_module_tests()
