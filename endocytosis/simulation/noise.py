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
import copy
import numpy as np

from ..contrib.PYME.Acquire.Hardware.Simulator import fakeCam
from ..config.camera import load_camera_yaml


class NoiseModel(object):
    """
    Parameters
    -----------
    path: str
    Path to the (YAML) camera metadata file.

    camera_serial_number: str
    Serial number of the camera whose metadata we'd like to use.
    """
    def __init__(self, path, camera_serial_number):
        self.path = path
        self.camera_serial_number = camera_serial_number
        self._data = load_camera_yaml(path, camera_serial_number)

    def load_camera_metadata(self, EMGainOn, readout_rate, preamp_setting):
        spec = self._data['specs']
        if EMGainOn:
            gain = 'EM Gain On'
        else:
            gain = 'EM Gain Off'
        readout_rate = str(int(readout_rate)) + 'MHz'
        preamp_setting = int(preamp_setting)
        spec = spec[gain][readout_rate][preamp_setting]
        self.gain = spec['TrueEMGain']
        self.read_noise = spec['readoutNoise']
        self.electrons_per_count = spec['electronsPerCount']

    def render(self, im):
        n_electrons = self.gain*2*np.random.poisson((im)/2)
        return (n_electrons +
                self.read_noise * np.random.standard_normal(n_electrons.shape)
                )/self.electrons_per_count
