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
from fijitools.helpers.coordinate import Coordinate

from endocytosis.simulation.psfmodel import SimpleGaussian2D
from endocytosis.simulation.image_components import FieldOfView
from endocytosis.simulation.noise import NoiseModel
from endocytosis.test import run


class TestFieldOfView(unittest.TestCase):
    def setUp(self):
        self.psf = SimpleGaussian2D(3., 6., 4.)

    def make_fov(self, noise):
        return FieldOfView((100, 100), 80., self.psf, noise)

    def make_noise_model(self):
        path = os.path.join(os.path.dirname(__file__), 'data', 'camera.yaml')
        nm = NoiseModel(path, 'X-9309')
        nm.load_camera_metadata(True, 30, 2)
        return nm

    def make_spots(self, fov):
        spot0 = fov.add_spot(Coordinate(nm=(3000, 3500)), 100., shape=(32, 32))
        spot1 = fov.add_spot(Coordinate(nm=(500, 500)), 100., shape=(32, 32))
        spot2 = fov.add_spot(Coordinate(nm=(7950, 7950)), 100., shape=(30, 30))
        spot3 = fov.add_spot(Coordinate(nm=(7950, 500)), 100., shape=(36, 36))
        spot4 = fov.add_spot(Coordinate(nm=(500, 7950)), 100., shape=(28, 28))
        return set([spot0, spot1, spot2, spot3, spot4])

    def test_create(self):
        # nn = 'no noise'
        nn = self.make_fov(None)
        im = nn.render()
        self.assertFalse(np.any(im))

        # wn = 'with noise'
        noise = self.make_noise_model()
        wn = self.make_fov(noise)
        im = wn.render()
        self.assertTrue(wn.noise)
        self.assertTrue(noise.electrons_per_count)
        self.assertTrue(np.any(noise.render(np.zeros((5, 5)))))
        self.assertTrue(np.any(im))

    def test_add_spot(self):
        fov = self.make_fov(None)
        spots = self.make_spots(fov)
        self.assertSetEqual(fov.children, spots)

    def test_render_with_spots(self):
        fov = self.make_fov(None)
        _ = self.make_spots(fov)
        _ = fov.render()
        self.assertFalse(fov.children.added)


# add this to all test modules
TESTS = [TestFieldOfView, ]


def run_module_tests():
    run(TESTS)

if __name__ == '__main__':
    run_module_tests()
