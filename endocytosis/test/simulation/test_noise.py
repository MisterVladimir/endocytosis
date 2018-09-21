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
import unittest
import numpy as np
import os

from ...simulation.noise import NoiseModel
from .. import run

__all__ = ['EMCCDNoiseModelTest']


class EMCCDNoiseModelTest(unittest.TestCase):
    def setUp(self):
        path = os.path.join(os.path.dirname(__file__), 'data', 'camera.yaml')
        self.im = np.random.rand((5, 5)) * 100.
        self.model = NoiseModel(path, 'X-7291')

    def test_render(self):
        self.model.render(self.im)


# add this to all test modules
TESTS = [EMCCDNoiseModelTest, ]


def run_module_tests():
    run(TESTS)

if __name__ == '__main__':
    run_module_tests()
