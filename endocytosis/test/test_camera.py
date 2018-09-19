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
from ruamel import yaml
import os

from endocytosis.config.camera import YAMLCameraSpec, make_YAML

__all__ = ['CameraSettingsTest']

CAMERA_SETTINGS = YAMLCameraSpec()
CAMERA_SETTINGS.add_specs(electronsPerCount=15., readoutNoise=250.,
                          TrueEMGain=10.)
CAMERA_SETTINGS.add_specs(electronsPerCount=11., readoutNoise=150.,
                          TrueEMGain=12., EM='EM Gain On',
                          readout_rate='17MHz', preamp_setting=3)
CAMERA_SETTINGS.add_specs(electronsPerCount=12., readoutNoise=200.,
                          TrueEMGain=9., EM='EM Gain On',
                          readout_rate='10MHz', preamp_setting=3)


class CameraSettingsTest(unittest.TestCase):
    path = 'data.camera_spec.yaml'

    def setUp(self):
        self.y = make_YAML()

    def test_instantiate(self):
        self.assertFalse(YAMLCameraSpec())

    def test_add_spec(self):
        cs = YAMLCameraSpec()
        cs.add_specs(electronsPerCount=15., readoutNoise=250., TrueEMGain=10.)
        cs.add_specs(electronsPerCount=11., readoutNoise=150., TrueEMGain=12.,
                     EM='EM Gain On', readout_rate='17MHz', preamp_setting=3)
        cs.add_specs(electronsPerCount=12., readoutNoise=200., TrueEMGain=9.,
                     EM='EM Gain On', readout_rate='10MHz', preamp_setting=3)

    def test_load(self):
        path = os.path.abspath(self.path)
        d2 = None
        with open(path, 'r') as f:
            d2 = self.y.load(f)
        self.assertIsInstance(d2, YAMLCameraSpec)

    def check_equality(self, d1, d2):
        self.assertDictEqual(d1, d2)
        for k1 in d1:
            self.assertIn(k1, d2)
            self.assertDictEqual(d1[k1], d2[k1])
            for k2 in d1[k1]:
                self.assertIn(k2, d2[k1])
                self.assertDictEqual(d1[k1][k2], d2[k1][k2])
                for k3 in d1[k1][k2]:
                    self.assertIn(k3, d2[k1][k2])
                    self.assertDictEqual(d1[k1][k2][k3],
                                         d2[k1][k2][k3])

    def test_equal(self):
        path = os.path.abspath(self.path)
        d1 = CAMERA_SETTINGS
        d2 = None
        with open(path, 'r') as f:
            d2 = self.y.load(f)
        self.check_equality(d1, d2)

    def test_dump(self):
        path = os.path.abspath('test.yaml')
        with open(path) as f:
            self.y.dump(CAMERA_SETTINGS, f)

        d2 = None
        with open(path, 'r') as f:
            d2 = self.y.load(f)
        self.assertIsInstance(d2, YAMLCameraSpec)
        self.check_equality(CAMERA_SETTINGS, d2)

        if os.path.isfile(path):
            os.remove(path)


class CameraTest(unittest.TestCase):
    def setUp(self):
        pass
