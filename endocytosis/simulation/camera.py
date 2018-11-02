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
from ruamel import yaml

from vladutils.io.yaml import YAMLDict


class YAMLCameraSpec(YAMLDict):
    yaml_tag = '!YAMLCameraSpec'

    def __init__(self):
        super().__init__()

    def __missing__(self, name):
        return YAMLDict(__parent=self, __key=name)

    def add_specs(self, electronsPerCount, readoutNoise, TrueEMGain,
                  EM='EM Gain On', readout_rate='17MHz', preamp_setting=1):
        pas = YAMLDict({'electronsPerCount': electronsPerCount,
                        'readoutNoise': readoutNoise,
                        'TrueEMGain': TrueEMGain})
        args = EM, readout_rate, preamp_setting
        args = tuple(map(str, args))
        self[EM][readout_rate][preamp_setting] = pas

    @classmethod
    def load(cls, constructor, node):
        constructor.flatten_mapping(node)
        d = constructor.construct_mapping(node, deep=True)
        ret = cls()
        for em in d.keys():
            for rr in d[em].keys():
                for pas, val in d[em][rr].items():
                    ret.add_specs(val['electronsPerCount'],
                                  val['readoutNoise'],
                                  val['TrueEMGain'],
                                  em, rr, pas)
        return ret


class YAMLCamera(YAMLDict):
    yaml_tag = '!YAMLCamera'

    def __init__(self, serial_number, ADOffset, quantum_efficiency=0.87,
                 temperature=-70., numGainElements=592, specs=None):
        super().__init__()
        yd = YAMLDict({'serial_number': serial_number,
                       'ADOffset': ADOffset,
                       'quantum_efficiency': quantum_efficiency,
                       'temperature': temperature,
                       'numGainElements': numGainElements,
                       'vbreakdown': 6.6})
        if specs:
            yd['specs'] = specs
        self[serial_number] = yd

    def __missing__(self, name):
        return YAMLDict(__parent=self, __key=name)

    @classmethod
    def load(cls, constructor, node):
        # BUG: YAMLCameraSpec are saved as YAMLDict
        constructor.flatten_mapping(node)
        d = constructor.construct_mapping(node, deep=True)
        serial_number = list(d.keys())[0]
        keys = ['ADOffset', 'quantum_efficiency',
                'temperature', 'numGainElements', 'specs']
        d = d[serial_number]
        kwargs = {k: d[k] for k in keys}
        return cls(serial_number, **kwargs)

    def add_specs(self, specs):
        self[self.serial_number]['specs'] = specs


def _make_YAML(typ='safe'):
    y = yaml.YAML(typ=typ)
    for c in (YAMLCamera, YAMLCameraSpec, YAMLDict):
        y.register_class(c)
    return y


def load_camera_yaml(path, serial_number):
    """
    Load EMCCD camera metadata from a YAML file.
    """
    y = _make_YAML()
    with open(path, 'r') as f:
        # y.load_all returns a generator that yields yaml documents
        # each document contains a camera's metadata as a YAMLDict
        # (a nested dictionary) whose root node's key is the camera's
        # serial number
        for item in y.load_all(f):
            if serial_number in item:
                return item[serial_number]
        else:
            # raise an error if we've iterated through all the cameras, and
            # haven't found the camera specified by the serial_number
            # parameter
            raise IOError("Serial number {} was not found.".format(
                serial_number))


def dump_camera_yaml(path, cameras):
    y = _make_YAML()
    cameras = [cameras]
    with open(path, 'w') as f:
        y.dump_all(cameras, f)
