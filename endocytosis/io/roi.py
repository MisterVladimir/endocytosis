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
import pathlib
import re
import zipfile
from os import listdir


class RoiReader(object):
    """
    """
    regexp = {'csv': r'.*\.{1}csv(\.json){0,1}', 'zip': '.*.zip'}
    available_file_formats = {'csv': '.csv', 'zip': '.zip'}

    def __init__(self):
        super().__init__()
        self.folder = None
        self.file_format = []
        self._base_glob = None

    def load_from_path(self, folder, file_format=None):
        # this isn't the most efficient algorithm, as basename filepaths are
        # first globbed, and then matched to the extension's regular expression
        self.folder = folder
        aff = self.available_file_formats

        if file_format is None:
            file_format == aff.keys()
        elif isinstance(file_format, str):
            file_format = list(file_format)
        else:
            assert np.iterable(file_format),
            "file_format must be a string interable containing the"
            "file_format(s)"

        for k in file_format:
            if k == 'csv':
                self._load_csv(folder)
            elif k == 'zip':
                self._load_zip(folder)

    def _load_zip(self, paths):
        regexp = re.compile(self.regexp['zip'])
        return [RoiZIP(p) for p in paths if regexp.match(p)]

    def _load_csv(self, folder):
        paths = listdir(folder)
        # find all CSV files that have metadata
        # metadata is in json format
        regexp = re.compile(self.regexp('csv'))
        paths = [p for p in paths if regexp.match(p)]
        sorted(paths)
        li = []
        paths = iter(paths)
        n = next(paths)
        for p in paths:
            if p == n[:-4]:
                li.append(RoiCSV(p, n))
                n = next(paths)
            elif n == p[:-4]:
                li.append(RoiCSV(n, p))
                n = next(paths)
        return li

    def load_from_file(self, file):
        pass


class RoiZIP(ROI):
    def __init__(self, path):
        super().__init__()
        self.data, self.metadata = self._read_zip(path)

    def _read_zip(self, path):
        
        return df


class RoiCSV(ROI):
    """
    """
    def __init__(self, path, md_path):
        super().__init__()
        self.data = self._read_csv(path)
        self.metadata = self._read_md(metadata)

    def _read_csv(self, path):
        # read csv and convert to ROI information
        return df

    def _read_md(self, md):
        # read json, and return as a nested dictionary
        return d


class ROI(object):
    """
    Contains roi and metadata about the source image.
    """
    pass