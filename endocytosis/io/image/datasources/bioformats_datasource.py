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
import javabridge
import bioformats

from endocytosis.io.datasources.base_datasource import (BaseImageRequest,
                                                        BaseDataSource)


class BioformatsRequest(BaseImageRequest):
    module_name = 'bioformats_datasource'

    def __init__(self, order, *shape, rescale=False):
        super().__init__(order, *shape)
        self.rescale = rescale

    def __call__(self, *ctzxy, series=0):
        if len(ctzxy) == 5:
            self['CTZXY'] = ctzxy
        elif len(ctzxy) == 3:
            ctzxy = list(ctzxy) + [slice(None), slice(None)]
            self['CTZXY'] = ctzxy
        elif len(ctzxy) == 0:
            # use previous contents of self
            pass
        else:
            raise Exception('')
        xsl, ysl = tuple(self['X']), tuple(self['Y'])
        x0, dx, y0, dy = xsl[0], xsl[1] - xsl[0], xsl[0], ysl[1] - ysl[0]
        return {'C': self[0], 'T': self[1], 'Z': [2], 'series': series,
                'rescale': self.rescale, 'XWYH': (x0, y0, dx, dy)}


class BioformatsDataSource(BaseDataSource):
    module_name = 'bioformats_datasource'

    def __init__(self, path, request):
        super().__init__()
        javabridge.start_vm(class_path=bioformats.JARS)
        self._request = request
        self.reader = bioformats.ImageReader(path)

    def request(self, *ctzxy, series=0):
        """
        Arguments
        -----------
        request : ImageDataRequest or list
        Each item in the list is a slice or int corresponding to the
        (channel, time, axial position) in that order.
        """
        kwargs = self._request(*ctzxy, series=series)
        return self.file.read(**kwargs)

    def cleanup(self):
        self.reader.close()
        javabridge.kill_vm()
