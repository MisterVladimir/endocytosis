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
from addict import Dict
from ruamel import yaml
import os

from ..helpers.data_structures import YAMLDict


def load_config(filename):
    y = yaml.YAML(typ='safe')
    y.register_class(YAMLDict)
    cfg_from_yaml = None
    with open(filename, 'r') as f:
        cfg_from_yaml = y.load(f)
    return cfg_from_yaml

path = os.path.join(os.path.dirname(__file__), 'config.yaml')
CONFIG = load_config(path)
