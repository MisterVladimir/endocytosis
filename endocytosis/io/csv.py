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

from endocytosis.io import IO


class CSVMetaData(dict):
    """
    Holds CSV file's metadata.
    """
    def __init__(self, data):
        super().__init__()


class CSVData(object):
    pass


class CSVReader(object):
    """
    Summary.
    """
    def __init__(self, path):
        super().__init__()
        self._repath = path
        self.load_from_path(path)

    def load_from_path(self, path):
        