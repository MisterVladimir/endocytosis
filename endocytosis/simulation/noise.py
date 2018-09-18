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
from abc import ABC, abstractmethod

from endocytosis.contrib.PYME.Acquire.Hardware.Simulator import fakeCam
from endocytosis.config import DEFAULT as cfg


class NoiseModel(fakeCam.NoiseModel):
    """
    Parameters
    -----------
    camera_serial_number: str
    Camera data should be set in 
    """
    def __init__(self, camera_serial_number):
        kwargs = cfg.CAMERA[camera_serial_number]
        super().__init__(**kwargs)

    def render(self, im):
        return self.noisify(im)
