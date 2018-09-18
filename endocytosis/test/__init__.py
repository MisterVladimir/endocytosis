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
import argparse
import os
from addict import Dict

from test_psfmodel import *
from test_noise import *

# 
MODULE_PATH_TO_TEST_NAME = Dict()
MODULE_PATH_TO_TEST_NAME.simulation.noise.EMCCDNoiseModel = EMCCDNoiseModelTest


def get_tests(names):
    """
    Returns the test classes associated with a package directory.

    Parameters
    -----------
    node: Dict
    """
    def recurse_dict_values(node):
        """
        Returns all the values in the nested dictionary.
        """
        if isinstance(node, dict):
            li = []
            for val in node.values():
                li += recurse_dict_values(val)
            return li
        elif isinstance(node, unittest.TestCase):
            return [node]
        else:
            # pass
            raise AssertionError('Not a TestCase.')

    def path_to_dict(li, dic):
        """
        Iterate through list of node names, and return the final
        node.

        Parameters
        -----------
        li: list or tuple of strings
        dic: Dict
        """
        try:
            name = li[0]
            if dic[name]:
                return path_to_dict(li[1:], dic[name])
        except IndexError:
            # list ended
            return dic[name]
        else:
            # last key look-up returned an empty Dict, which means
            # that key was not present in the Dict
            return False

    tests = []
    for name in names:
        name = name.split('.')
        res = path_to_dict(name, MODULE_PATH_TO_TEST_NAME)
        if res:
            tests += recurse_dict_values(res)

    return tests

MODULE_NAMES_HELP = ('Input the path of the item you would '
                     'like to test, e.g. endocytosis.simulation.noise')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--names', nargs='+', type=str,
                        required=False, help=MODULE_NAMES_HELP)
    args = parser.parse_args()

    test_classes = get_tests(args.names)
    loader = unittest.TestLoader()
    runner = unittest.TextTestRunner(verbosity=2)
    for class_ in test_classes:
        loaded_tests = loader.loadTestsFromTestCase(class_)
        runner.run(loaded_tests)
