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
from os.path import relpath, sep, isfile, splitext, isdir
from addict import Dict
import glob
import importlib
import os


def get_module_names(path):
    """
    Recursively get all python modules (besides __init__) in path.
    Returns module name as relative to path.
    """
    def recurse(p):
        result = []
        li = glob.glob(p + sep + '*')
        dirs = [_p for _p in li if isdir(_p)]
        files = \
            [splitext(relpath(_p, path))[0].replace(sep, '.')
             for _p in li if isfile(_p) and _p.endswith('.py') and
             not _p.endswith('__init__.py')]

        for d in dirs:
            result += recurse(d) + files
        return result

    return recurse(path)


def get_tests(li):
    tests = [getattr(importlib.import_module(n), 'TESTS') for n in li]
    ret = []
    for t in tests:
        ret += t
    return ret


MODULE_NAMES_HELP = ('Input the path of the item you would '
                     'like to test, e.g. endocytosis.simulation.noise')


def run(tests):
    """
    Run a test.

    Paramters
    ------------
    tests: iterable
    Contains the unittest.TestCase to be run.
    """
    loader = unittest.TestLoader()
    runner = unittest.TextTestRunner(verbosity=2)
    for t in tests:
        loaded = loader.loadTestsFromTestCase(t)
        runner.run(loaded)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--names', nargs='+',
                        required=False, help=MODULE_NAMES_HELP)
    args = parser.parse_args()

    names = args.names
    if not names:
        names = os.path.abspath(os.path.dirname(__file__))

    module_names = get_module_names(names)
    tests = get_tests(names)
    run(tests)
