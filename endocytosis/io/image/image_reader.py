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
from abc import ABC, ABCMeta
import h5py
from os.path import abspath
from copy import copy
from scipy.optimize import minimize
import pickle
import tensorflow as tf
import bioformats

from endocytosis.helpers.data_structures import TrackedList
from endocytosis.helpers.coordinate import Coordinate
from endocytosis.contrib.gohlke import tifffile
from endocytosis.io import IO


class ImageReader(IO):
    def __init__(self):
        super().__init__()

    def load(self, path=None, data=None, metadata=None):
        if path is not None:
            self._load_path(path)

    def _load_path(self, path):
        pass
