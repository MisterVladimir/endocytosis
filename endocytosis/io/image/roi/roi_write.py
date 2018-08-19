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
import pandas as pd
import re
import zipfile
import os
from struct import pack, unpack

from endocytosis.helpers.data_structures import ListDict
from endocytosis.io import IO
from endocytosis.io.image.roi import (HEADER_SIZE, HEADER2_SIZE,
                                      HEADER_DTYPE, HEADER2_DTYPE,
                                      OPTIONS, SUBTYPE, ROI_TYPE)
from endocytosis.io.image.roi.roi_objects import ROI


class Writer(IO):
    pass


class IJZipWriter(Writer):
    """
    The goal is to look like a container for Fiji ROI extracted from the
    input path.

    Inspiration from :
    1) http://grepcode.com/file/repo1.maven.org/maven2/gov.nih.imagej/imagej/1.47/ij/io/RoiDecoder.java 
    2) http://grepcode.com/file/repo1.maven.org/maven2/gov.nih.imagej/imagej/1.47/ij/io/RoiEncoder.java 
    3) https://github.com/hadim/read-roi/ 
    4) https://github.com/DylanMuir/ReadImageJROI

    See also:
    https://github.com/imagej/ImageJA/blob/master/src/main/java/ij/io/RoiEncoder.java and
    https://github.com/imagej/ImageJA/blob/master/src/main/java/ij/io/RoiDecoder.java

    Parameters
    -----------
    placeholder
    """

    def __init__(self, zip_path):
        self.zip_path = zip_path
        self._file = zipfile.ZipFile(zip_path, 'a')

    def write(self, roi, roi_name, image_name=''):
        """
        """
        data = roi.to_IJ(roi_name, image_name)
        with self._file.open(roi_name + ".roi", 'a') as f:
            f.write(data)

    def cleanup(self):
        self._file.close()


class CSVWriter(Writer):
    pass
