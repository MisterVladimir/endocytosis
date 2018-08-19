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
from struct import unpack
from collections import OrderedDict

from endocytosis.helpers.data_structures import ListDict
from endocytosis.io import IO
from endocytosis.io.image.roi import (HEADER_SIZE, HEADER2_SIZE,
                                      HEADER_DTYPE, HEADER2_DTYPE,
                                      COLOR_DTYPE, OPTIONS,
                                      SUBTYPE, ROI_TYPE, COLOR_DTYPE, SELECT_ROI_PARAMS)
from endocytosis.io.image.roi.roi_objects import ROI


class Reader(IO):
    pass


class IJZipReader(Reader):
    """
    The goal is to convert ImageJ/FIJI ROI bytestreams
    to human-readable data using python. Unlike other libraries
    I'm aware of, here we let numpy do the heavy lifting of
    converting bytestream to bytes, shorts, integers, and floats.
    Strings, e.g. the ROI name, are unpacked using the struct
    library. This information is stored as a numpy.recarray.
    See global variables imported from roi.__init__ to understand
    which position in the bytestream corresponds to which ROI
    parameters.

    Inspiration from:
    1) http://grepcode.com/file/repo1.maven.org/maven2/gov.nih.imagej/imagej/1.47/ij/io/RoiDecoder.java 
    2) http://grepcode.com/file/repo1.maven.org/maven2/gov.nih.imagej/imagej/1.47/ij/io/RoiEncoder.java 
    3) https://github.com/hadim/read-roi/
    4) https://github.com/DylanMuir/ReadImageJROI

    More recent versions of the Java code:
    https://github.com/imagej/ImageJA/blob/master/src/main/java/ij/io/RoiEncoder.java and
    https://github.com/imagej/ImageJA/blob/master/src/main/java/ij/io/RoiDecoder.java

    Parameters
    -----------
    regexp: str
    Argument for re.compile() to filter roi names within the zip file.

    sep: str
    ROI data may contain classes of ROI, for example those labeling biological
    structure. As many instances of one class may be found in one image, the
    user may specify classes in the ROI name according to the following format:
    [class][sep][integer]. If sep is not None -- in which every ROI is read as
    a distinct class -- then IJRoiDecoder stores ROI data in nested dictionary
    format.
    """

    def __init__(self, regexp='.*roi$', sep=None):
        self.regexp = re.compile(regexp)
        self.sep = sep
        self._data = ListDict()

    def __getitem__(self, key):
        """
        """
        return self._data[key]

    def __add__(self, other):
        """
        To combine ROI lists together, or append .roi.ROI objects
        """
        return NotImplemented

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def read(self, path, pwd=None):
        # clear old data
        self._data = ListDict()
        # reads all the zip files' byte streams, sends them to parsing function 
        self._file = zipfile.ZipFile(path, 'r')
        filelist = [f for f in self._file.namelist() if self.regexp.match(f)]
        streams = [None]*len(filelist)
        for i, name in enumerate(filelist, 'r'):
            with self.file.open(name) as f:
                streams[i] = f.read()

        # file type checking: .roi files' first four bytes encode 'Iout'
        self.bytestreams = [s for s in streams if s[:4] == b'Iout']
        self._parse_bytestream()

    def _parse_bytestream(self):
        """
        Pass in a list of byte data (bs=bytestream); turn them into numpy
        arrays.

        Most of the info in the bytestream isn't relevant to my needs, but it
        should be straightforward to add it to self._data.
        """
        # parse header data
        length = len(self.bytestreams)
        hdr = np.recarray(length, HEADER_DTYPE, buf=b''.join([
            b[:HEADER_SIZE] for b in self.bytestreams]))

        hdr2_offsets = hdr['hdr2_offset']
        hdr2 = np.recarray(length, HEADER2_DTYPE, b''.join([
            b[h:h+HEADER2_SIZE] for h, b in zip(hdr2_offsets,
                                                self.bytestreams)]))

        # no need to distinguish between 'oval' and 'ellipse'
        # the only difference is the latter always has subpixel resolution
        is_ellipse = hdr['subtype'] == SUBTYPE['ellipse']
        hdr['type'][is_ellipse] = ROI_TYPE['oval']

        name_offsets = hdr2['name_offset']
        name_lengths = hdr2['name_length']
        # assumes characters are in ascii encoding...
        names = self._get_names(name_offsets, name_lengths)
        # determine if subpixel resolution
        subpixel = np.logical_and(hdr['options'] //
                                  OPTIONS['sub_pixel_resolution'] > 0,
                                  hdr['version'] >= 222)
        # parse parameters common to all ROI
        bounding_rect, common = self._get_common(hdr, hdr2, subpixel)
        points = self._get_points(hdr, subpixel)
        props = self._get_roi_props(hdr['hdr2_offsets'],
                                    hdr2['roi_props_offsets'],
                                    hdr2['roi_props_lengths'])

        for br, com, p, pr, typ, name in zip(bounding_rect, common, points,
                                             props, hdr['type'], names):
            if name[1]:
                self._data[name[0]].append(ROI(br, com, p, pr, typ))
            else:
                self._data[name[0]] = ROI(br, com, p, pr, typ)

    def _get_names(self, offsets, lengths):
        """
        Decode roi names from the bytestream.
        """
        # assumes characters are in ascii encoding...
        names = ["".join(map(chr, unpack('>'+'h'*le, bs[off:off+le*2]))) for
                 bs, off, le in zip(self.bytestreams, offsets, lengths)]
        # names = [bs[off:off+le*2].decode('utf-8') for bs, off, le in zip(
        #          self.bytestreams, offsets, lengths)]
        if self.sep:
            names = [name.split(self.sep) for name in names]
            return [li + [''] if len(li) == 1 else li for li in names]
        else:
            return [[n, ''] for n in names]

    def _get_common(self, hdr, hdr2, subpixel):
        coord_dtype = [('x1', float), ('y1', float),
                       ('x2', float), ('y2', float)]
        coords = np.zeros(len(hdr), dtype=coord_dtype)
        for a, b, c in zip(['y1', 'x1', 'y2', 'x2'],
                           ['x1', 'y1', 'x2', 'y2'],
                           ['top', 'left', 'bottom', 'right']):
            # if subpixel, use 'x1', 'y1', 'x2', 'y2'
            # otherwise use 'top', 'left', 'bottom', 'right'
            # reverses x and y axes of ImageJ
            coords[a][subpixel] = hdr[b][subpixel]
            coords[a][~subpixel] = hdr[c][~subpixel]

        d = OrderedDict(**SELECT_ROI_PARAMS['hdr'],
                        **SELECT_ROI_PARAMS['hdr2'])
        common_dtype = np.dtype(dict(names=list(d.keys()),
                                     formats=list(d.values())))
        common = np.zeros(len(hdr), dtype=common_dtype)
        keys = list(SELECT_ROI_PARAMS['hdr'].keys())
        common[keys] = hdr[keys]
        keys2 = list(SELECT_ROI_PARAMS['hdr2'].keys())
        common[keys2] = hdr2[keys2]

        return coords, common

    def _get_points(self, hdr, subpixel):
        # multi-point and individual points not yet implemented,
        # but in the works
        types = [ROI_TYPE['polygon'], ROI_TYPE['freeline'],
                 ROI_TYPE['polyline'], ROI_TYPE['freehand']]
        type_mask = np.isin(hdr['type'], types)
        size = HEADER_SIZE
        coords = [[unpack('>'+n*'f', bs[size+8*n:size+12*n]),
                   unpack('>'+n*'f', bs[size+4*n:size+8*n])]
                  if s else [unpack('>'+n*'h', bs[size+2*n:size+4*n]) + le,
                             unpack('>'+n*'h', bs[size:size+2*n]) + to]
                  for n, to, le, s, bs in zip(hdr['n_coordinates'],
                                              hdr['top'],
                                              hdr['left'],
                                              subpixel,
                                              self.bytestreams)]
        return np.array([(c[0], c[1]) if i else ([], []) for c, i in
                         zip(coords, type_mask)]).T

    def _get_roi_props(self, hdr2_offsets, roi_props_offsets,
                       roi_props_lengths):
        return [''.join(list(map(chr, unpack('>' + 'h'*le, bs[off:off+le*2]))))
                for le, bs, off in zip(
                roi_props_lengths, self.bytestreams, roi_props_lengths)]

    def _get_point_counters(self, offsets, n_coordinates):
        """
        This helps parse a multi-point roi.

        Point counters are ints that encode two parameters: 'positions' and
        'counters'. They describe which c, t, z slice the point roi lies in
        and the index of the point roi in the set, respectively. 'Position'
        is encoded as a short in byte positions 1 and 2, and 'counter' is
        a byte in position 3. Byte 0 is not used.

        Arguments
        -----------
        offset: int
        hdr['counters_offset']

        n_coordinates: int
        hdr['n_coordinates']
        """
        data = [unpack('>'+'hbb'*(n-1) + 'hb', bs[off+1:off+n*4])
                for n, bs, off in zip(
                n_coordinates, self.bytestreams, offsets)]
        positions = [p[::3] for p in data]
        counters = [c[1::3] for c in data]
        return counters, positions

    def cleanup(self):
        self._file.close()


class CSVReader(Reader):
    """
    Opens a CSV file and the associated metadata file, converts it to ROI
    format.
    """
    def __init__(self, path, metadata):
        super().__init__()
        self.data = self._read_csv(path)
        self.metadata = self._read_metadata(metadata)

    def _read_csv(self, path):
        # read csv and convert to ROI information
        return NotImplemented

    def _read_metadata(self, md):
        # read json, and return as a nested dictionary
        return NotImplemented
