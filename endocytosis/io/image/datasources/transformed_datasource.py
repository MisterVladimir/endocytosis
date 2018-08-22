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
import cv2 as cv
from numpy import random

from endocytosis.io.image.datasources import NestedDatasource
from endocytosis.helpers.errors import DatasourceError


class TransformedDatasource(NestedDatasource):
    """

    Parameters
    ------------
    datasource: NestedDatasource, FileDataSource
    Image data or 

    output_size: numpy.ndarray
    Optional. Shape of the output image. If parameter is left as 'same', do not
    resize.
    
    random_crop: numpy.ndarray
    Optional. If we want to take a random crop along the image's XY dimensions,
    enter the shape of the cropped image. Cropping is done before the image is
    resized.
    """

    def __init__(self, data, output_size='same', random_crop=None):
        super().__init__(data)
        self.datasource = data.datasource
        self.output_size = output_size
        self.random_crop = random_crop

    def request(self, *ctzxy):
        im = self.datasource.request(*ctzxy)
        if self.random_crop:
            max_start = im.shape - self.random_crop
            x0, y0 = [random.randint(0, i) for i in max_start]
            x1, y1 = [x0, y0] + self.random_crop
            im = im[x0:x1, y0:y1]

        if not self.output_size == 'same':
            im = cv.resize(im, self.output_size, interpolation=cv.INTER_CUBIC)
        return im
