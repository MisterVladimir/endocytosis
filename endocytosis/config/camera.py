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
from copy import copy


def addict_representer(dumper, data):
    def as_typical_dict(addict_dict):
        """
        Returns an addict.Dict as a dict.
        """
        if isinstance(addict_dict, Dict):
            d = {}
            for k, v in addict_dict.items():
                d[k] = as_typical_dict(v)
            return d
        else:
            return addict_dict

    data = as_typical_dict(data)
    return dumper.represent_dict(data)


class CameraSetting(yaml.YAMLObject):
    yaml_tag = '!CameraSetting'

    def __init__(self, serial_number, ADOffset, quantum_efficiency,
                 temperature, specs):
        super().__init__()
        self.serial_number = str(serial_number)
        self.ADOffset = ADOffset
        self.quantum_efficiency = quantum_efficiency
        self.temperature = temperature
        self.specs = specs
        self.vbreakdown = 6.6

    def add_specs(self, specs, EM='EM Gain On', readout_rate='17MHz',
                  preamp_setting=1):
        EM, readout_rate, preamp_setting = map(str, (EM, readout_rate,
                                                     preamp_setting))
        self.specs[EM][readout_rate][preamp_setting] = specs

    def __repr__(self):
        d = Dict()
        d.serial_number = self.serial_number
        d.ADOffset = self.ADOffset
        d.quantum_efficiency = self.quantum_efficiency
        d.temperature = self.temperature
        d.vbreakdown = self.vbreakdown
        d.specs = self.specs
        return d.__repr__()


def load_camera(path):
    ret = None
    with open(path, 'r') as f:
        ret = yaml.load_all(f.read())
    return ret


def write_camera(path, cameras):
    dumper = yaml.Dumper()
    with open(path, 'w') as f:
        