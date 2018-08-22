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


# copied from https://docs.python.org/3.6/tutorial/errors.html
class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class DatasourceError(Error):
    """
    Exception raised for errors in the input.

    Attributes
    -------------
    expression: str
    Input expression in which the error occurred.

    message: str
    Explanation of the error
    """

    def __init__(self, message):
        # self.expression = expression
        self.message = message
        print(self.message)
