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
import os
from ruamel import yaml
from vladutils.io.yaml import YAMLDict

FOLDER = os.path.dirname(__file__)
FILENAME = os.path.join(FOLDER, 'config.yaml')


def reload_config(filename=FILENAME):
    y = yaml.YAML(typ='safe')
    y.register_class(YAMLDict)
    cfg_from_yaml = None
    with open(filename, 'r') as f:
        cfg_from_yaml = y.load(f)
    return cfg_from_yaml

CONFIG = reload_config()


def reset_global_config():
    globals()['CONFIG'] = reload_config()

__all__ = ['CONFIG', 'reload_config', 'reset_global_config']
