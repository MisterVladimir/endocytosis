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

from endocytosis.simulation.psfmodel import SimpleGaussian2D
from endocytosis.test import run


class TestSimpleGaussian2D(unittest.TestCase):
    def setUp(self):
        self.psf = SimpleGaussian2D(2., 5., 6.)

    def test_render(self):
        self.psf.render(100, (16, 16))


# add this to all test modules
TESTS = [TestSimpleGaussian2D, ]


def run_module_tests():
    run(TESTS)

if __name__ == '__main__':
    run_module_tests()
