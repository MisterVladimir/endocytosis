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

from endocytosis.io import IO
from endocytosis.io.image import metadata


class ImageReader(IO):
    """
    Image input/output. Currently only works for tif images.
    """
    def __init__(self):
        super().__init__()

    def load(self, path=None, data=None, metadata=None):
        if path is not None:
            self._load_path(path)

    def _load_path(self, path):
        if path.endswith('.tif') or path.endswith('.tiff'):
            self._load_tif(path)
        else:
            raise TypeError('Not a compatible file type.')

    def _load_tif(self, path):
        from endocytosis.contrib.gohlke import tifffile
        from endocytosis.io.image.datasources import tiff_datasource
        # placeholder for more sophisticated metadata extraction
        is_imagej = False
        with tifffile.TiffFile(path) as tif:
            if tif.is_imagej:
                self.metadata = metadata.MetaData(tif.imagej_metadata,
                                                  'ImageJ',
                                                  tif.filename)
                is_imagej = True
            elif tif.is_ome:
                self.metadata = metadata.MetaData(tif.imagej_metadata,
                                                  'OME',
                                                  tif.filename)
            else:
                raise TypeError('')
        md = self.metadata
        shape = [md['SizeC'], md['SizeT'], md['SizeZ'],
                 md['SizeX'], md['SizeY']]
        shape = list(map(int, shape))
        if is_imagej:
            # not sure if 'YX' is correct
            request = tiff_datasource.TiffImageRequest('TZCYX', *shape)
        else:
            order = md['DimensionOrder']
            request = tiff_datasource.TiffImageRequest(order, *shape)
        self.data = tiff_datasource.TiffDataSource(path, request)

    def cleanup(self):
        try:
            self.data.cleanup()
        except AttributeError:
            pass