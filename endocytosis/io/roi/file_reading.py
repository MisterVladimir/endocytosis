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

from endocytosis.helpers.data_structures import ListDict
from endocytosis.io import IO
from endocytosis.io.roi import roi_type
from endocytosis.io.roi.roi_objects import ROI


class RoiPathFinder(object):
    """
    Parameters
    -----------
    regexp: str
    Regular expression argument for file name before to extension.

    extension: str
    Extension of desired.
    """
    extension_regexp = {'csv': r'.*\.{1}csv(\.json){0,1}', 'zip': '.*.zip'}
    # available_file_formats = {'csv': '.csv', 'zip': '.zip'}

    def __init__(self, regexp=r'.*', extension=None):
        self.extension = extension
        self.regexp = regexp

    def load(self, path):
        """
        Returns a list of 

        Parameters
        -----------
        path: str
        Folder or file name to load ROI data from.
        """
        # this isn't the most efficient algorithm, as basename filepaths are
        # first globbed, and then matched to the extension's regular expression
        ext = self.extension

        splitpath = path.split(os.path.extsep)
        # path is a file name
        if len(splitpath) > 1:
            self.folder = os.path.dirname(path) + os.path.pathsep
            if splitpath[-1] == 'csv':
                csv_path, metadata_path = self._get_csv_metadata_path(path)
                return [csv_path, metadata_path]
            elif splitpath[-1] == 'json' and splitpath[-2] == 'csv':
                csv_path, metadata_path = self._get_csv_metadata_path(
                    "{}".format(os.path.extsep).join(splitpath[:-1]))
                return [csv_path, metadata_path]

            elif splitpath[-1] == 'zip':
                return [os.path.basename(path)]

        # path is a path
        elif ext is not None:
            if ext == 'csv' or ext == '.csv':
                # csv files are a special case because in addition to getting
                # the csv file we must also get the associated metadata,
                # which is in JSON format
                return self._get_csv_filenames(path)
                # return [self._load_csv(*f) for f in filenames]
            elif ext == 'zip' or ext == '.zip':
                return self._get_filenames(path, 'zip')

        elif ext is None:
            raise TypeError('Please set extension for folder name arguments.')

        else:
            raise TypeError('File format of {} is not compatible. Please enter '
                            'a file or path name that contains {} '
                            'files.'.format(path, list(self.extension_regexp.keys())))

    def _get_csv_metadata_path(self, csv_path):
        return NotImplemented, NotImplemented

    def _get_csv_filenames(self, folder):
        # TODO: not tested
        # find all CSV files that have metadata
        regexp = re.compile(self.regexp + self.extension_regexp['csv'])
        filenames = [p for p in os.listdir(folder) if regexp.match(p)]
        sorted(filenames)
        li = []
        filenames = iter(filenames)
        n = next(filenames)
        for p in filenames:
            if p == n[:-4]:
                li.append((p, n))
                n = next(filenames)
            elif n == p[:-4]:
                li.append((n, p))
                n = next(filenames)
        return li

    def _get_filenames(self, folder, extension):
        regexp = re.compile(self.regexp + self.extension_regexp[extension])
        filenames = os.listdir(folder)
        return [p for p in filenames if regexp.match(p)]


class Reader(IO):
    pass


class IJZipReader(Reader):
    """
    The goal is to look like a container for Fiji ROI extracted from the
    input path.

    Inspiration from :
    1) http://grepcode.com/file/repo1.maven.org/maven2/gov.nih.imagej/imagej/1.47/ij/io/RoiDecoder.java 
    2) http://grepcode.com/file/repo1.maven.org/maven2/gov.nih.imagej/imagej/1.47/ij/io/RoiEncoder.java 
    3) https://github.com/hadim/read-roi/ 
    4) https://github.com/DylanMuir/ReadImageJROI

    Parameters
    -----------
    path: str
    Path to zip file containing ROI data.

    sep: str
    ROI data may contain classes of ROI, for example those labeling biological
    structure. As many instances of one class may be found in one image, the
    user may specify classes in the ROI name according to the following format:
    [class][sep][integer]. If sep is not None -- in which every ROI is read as
    a distinct class -- then IJRoiDecoder stores ROI data in nested dictionary
    format.

    image_name: str
    """
    # version 1.51
    header_size = 64
    # version 1.50??? 64 #version 1.51???
    header2_size = 40
    header_dtype = np.dtype(dict(
        names=['magic', 'version', 'type', 'top', 'left', 'bottom',
               'right', 'n_coordinates', 'x1', 'y1', 'x2', 'y2',
               'stroke_width', 'shape_roi_size', 'stroke_color',
               'fill_color', 'subtype', 'options', 'style_or_ratio',
               'arrow_head_size', 'rounded_rect_arc_size', 'position',
               'hdr2_offset'],
        offsets=[0, 4, 6, 8, 10, 12,
                 14, 16, 18, 22, 26, 30,
                 34, 36, 40,
                 44, 48, 50, 52,
                 53, 54, 56,
                 60],
        formats=[(bytes, 4), '>i2', '<i2', '>i2', '>i2', '>i2',
                 '>i2', '>i2', '>f4', '>f4', '>f4', '>f4',
                 '>i2', '>i2', '>i4',
                 '>i4', '>i2', '>i2', 'i1',
                 'i1', '>i2', '>i4',
                 '>i4']))

    header2_dtype = np.dtype(dict(
        names=['c', 'z', 't', 'name_offset', 'name_length', 'label_color',
               'font_size', 'opacity', 'image_size', 'float_stroke_width'],
        offsets=[4, 8, 12, 16, 20, 24,
                 28, 31, 32, 36],
        formats=['>i4', '>i4', '>i4', '>i4', '>i4', '>i4',
                 '>i2', 'i1', '>i4', '>i4']))
    options = {'spline_fit': 1,
               'double_headed': 2,
               'outline': 4,
               'overlay_labels': 8,
               'overlay_names': 16,
               'overlay_backgrounds': 32,
               'overlay_bold': 64,
               'sub_pixel_resolution': 128,
               'draw_offset': 256}

    def __init__(self, path, sep='-'):
        self.path = path
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
        pass

    def read(self):
        # reads all the zip files' byte streams, sends them to parsing function 
        streams = []
        self._file = zipfile.ZipFile(self.path, 'r')
        streams = [self._file.open(n).read() for n in self._file.namelist()]
        streams = [s for s in streams if s[:4] == b'Iout']
        self.bytestreams = streams
        self._parse_bytestream(streams)

    def _set_keys(self, names):
        self._original_names = names
        for name in names:
            _name = name.split(self.sep)
            if len(_name) == 1:
                self._data[name[0]] = None
            elif len(_name) == 2:
                self._data[_name[0]][_name[1]] = None

    def _parse_bytestream(self, bs):
        """
        Pass in a list of byte data (bs=bytestream); turn them into numpy
        arrays. 

        Most of the info in the bytestream isn't relevant to my needs, but it
        should be straightforward to add it to self._data.
        """
        # parse header data
        length = len(bs)
        hdr = np.recarray(length, self.header_dtype, buf=b''.join([
            b[:self.header_size] for b in bs]))

        hdr2_offsets = hdr['hdr2_offset']
        hdr2 = np.recarray(length, self.header2_dtype, b''.join([
            b[h:h+self.header2_size] for h, b in zip(hdr2_offsets, bs)]))

        self.hdr = hdr
        self.hdr2 = hdr2

        name_offsets = hdr2['name_offset']
        name_lengths = hdr2['name_length']
        # assumes characters are in ascii encoding...
        names = self._names_from_bytestream(name_offsets, name_lengths)
        # determine if subpixel resolution
        subpixel = np.logical_and(hdr['options'] //
                                  self.options['sub_pixel_resolution'] > 0,
                                  hdr['version'] >= 222)
        # parse parameters common to all ROI
        common = self._parse_common(hdr, hdr2, subpixel)
        points = self._parse_points(hdr, subpixel)

        for c, p, typ, name in zip(common, points, hdr['type'], names):
            if name[1]:
                self._data[name[0]].append(ROI(c, p, typ))
            else:
                self._data[name[0]] = ROI(c, p, typ)

    def _names_from_bytestream(self, offsets, lengths):
        # assumes characters are in ascii encoding...
        names = ["".join(map(chr, unpack('>'+'h'*le, bs[off:off+le*2]))) for
                 bs, off, le in zip(self.bytestreams, offsets, lengths)]
        # names = [bs[off:off+le*2].decode('utf-8') for bs, off, le in zip(
        #          self.bytestreams, offsets, lengths)]
        names = [name.split(self.sep) for name in names]
        return [li + [''] if len(li) == 1 else li for li in names]

    def _parse_common(self, hdr, hdr2, subpixel):
        dtype = [('x1', float), ('y1', float),
                 ('x2', float), ('y2', float),
                 ('c', int), ('t', int), ('z', int)]
        result = np.zeros(len(hdr), dtype=dtype)

        for a, b, c in zip(['y1', 'x1', 'y2', 'x2'],
                           ['x1', 'y1', 'x2', 'y2'],
                           ['top', 'left', 'bottom', 'right']):
            # if subpixel, use 'x1', 'y1', 'x2', 'y2'
            # otherwise use 'top', 'left', 'bottom', 'right'
            # reverses x and y axes of ImageJ
            result[a][subpixel] = hdr[b][subpixel]
            result[a][~subpixel] = hdr[c][~subpixel]

        for key in ['c', 't', 'z']:
            result[key] = hdr2[key]

        return result

    def _parse_points(self, hdr, subpixel):
        types = [roi_type['polygon'], roi_type['freeline'],
                 roi_type['polyline'], roi_type['freehand']]
        type_mask = np.isin(hdr['type'], types)
        size = self.header_size
        coords = [[unpack('>'+n*'f', bs[size+8*n:size+12*n]),
                   unpack('>'+n*'f', bs[size+4*n:size+8*n])]
                  if s else [unpack('>'+n*'h', bs[size+2*n:size+4*n]) + le,
                             unpack('>'+n*'h', bs[size:size+2*n]) + to]
                  for n, to, le, s, bs in zip(hdr['n_coordinates'],
                                              hdr['top'],
                                              hdr['left'],
                                              subpixel,
                                              self.bytestreams)]
        return [(c[0], c[1]) if i else ([], []) for c, i in
                zip(coords, type_mask)]


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
