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
from abc import ABC
from copy import copy


class Coordinate(ABC):
    _unit_conversion = {'m': 1., 'um': 1e-3, 'nm': 1e-6}

    def __len__(self):
        return len(self._parameters)

    def __ne__(self, other):
        return not self.__eq__(other)

    def convert_units(self, number, units):
        """
        Convert number with given units to self.units.

        number : int or float
        units: string
        One of 'nm', 'um', or 'm'.
        """
        if self.units == units:
            return number
        else:
            self_factor = self._unit_conversion[self.units]
            other_factor = self._unit_conversion[units]
            factor = self_factor / other_factor
            return number / factor


class Coordinate1D(Coordinate):
    _parameters = {'value': 0.}

    def __init__(self, value=0., units='nm', coordinate=None):
        super().__init__()
        if coordinate is None:
            # coerce units to nm
            self.value = value
            self._units = units
        elif isinstance(coordinate, type(self)):
            self.value = coordinate.value
            self._units = coordinate.units
        else:
            raise('')

    def __eq__(self, other):
        if isinstance(other, Coordinate1D):
            # convert my units to other's units
            other_value = self.convert_units(other.value, self.units)
            return self.value == other_value
        elif isinstance(other, (int, float)):
            return self.value == other
        else:
            return False

    def __add__(self, other):
        if isinstance(other, Coordinate1D):
            other_value = self.convert_units(other.value, self.units)
            return self.value + other_value
        elif isinstance(other, (int, float)):
            return self.value + other

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Coordinate1D(self.value * other, self.units)
        elif hasattr(other, __len__) and \
                len(other) == 1 and \
                not isinstance(other, (str, Coordinate)):
            return Coordinate1D(self.value * other[0], self.units)
        else:
            raise TypeError('Coordinates may only be multiplied by numbers or'
                            ' arrays.')

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Coordinate1D(self.value * other, self.units)
        elif hasattr(other, __len__) and \
                len(other) == 1 and \
                not isinstance(other, (str, Coordinate)):
            return Coordinate1D(self.value / other, self.units)
        else:
            raise TypeError('Coordinates may only be divided by numbers or'
                            ' arrays.')

        return Coordinate2D(x, y, 'nm')

    def __floordiv__(self, other):
        if isinstance(other, int):
            return Coordinate1D(self.value // other, self.units)
        else:
            return self.__truediv__(other)

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, name):
        if name not in self._unit_conversion.keys():
            raise("{} not supported.".format(name))
        self.value = self.convert_units(self.value, name)
        self._units = name

    @property
    def value(self):
        return self._parameters['value']

    @value.setter
    def value(self, val):
        self._parameters['value'] = val


class Coordinate2D(Coordinate):
    _parameters = {'x': None, 'y': None}

    def __init__(self, x=0., y=0., units='nm', coordinate=None):
        super().__init__()
        if coordinate is None:
            if isinstance(x, Coordinate1D) and isinstance(y, Coordinate1D):
                self.x = x
                self.y = y
            elif isinstance(x, Coordinate1D) and isinstance(y, (int, float)):
                self.x = x
                self.y = Coordinate1D(y, x.units)
            elif isinstance(y, Coordinate1D) and isinstance(x, (int, float)):
                self.y = y
                self.x = Coordinate1D(x, y.units)
            elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
                self.x = Coordinate1D(x, units)
                self.y = Coordinate1D(y, units)
            else:
                raise TypeError('x and y must be ints, floats or Coordinate1D '
                                'objects.')
        elif isinstance(coordinate, Coordinate2D):
            self.x = coordinate.x
            self.y = coordinate.y
        else:
            raise('')

    @property
    def units(self):
        if self.x.units == self.y.units == self._units:
            return self._units
        else:
            self.y.units = self._units
            self.x.units = self._units
            return self._units

    @units.setter
    def units(self, name):
        if name not in self._unit_conversion.keys():
            raise("{} not supported unit.".format(name))
        for k, v in self._parameters.items():
            self.__dict__[k].units = name
        self._units = name

    @property
    def x(self):
        return self._parameters['x']

    @x.setter
    def x(self, value):
        self._parameters['x'] = value

    @property
    def y(self):
        return self._parameters['y']

    @y.setter
    def y(self, value):
        self._parameters['y'] = value

    @property
    def asarray(self):
        return np.array((self.x.value, self.y.value))

    @property
    def as_structured_array(self):
        return np.array(
            (self.x.value, self.y.value), dtype=(('x', '<f4'), ('y', '<f4')))

    def to_pixels(self, pixelsize):
        return self.asarray / pixelsize

    def __eq__(self, other):
        try:
            if len(other) == len(self) and not isinstance(
                    other, (str)):
                if isinstance(other, Coordinate2D):
                    # convert my units to other's units
                    other_x = self.convert_units(other.x.value, other.x.units)
                    other_y = self.convert_units(other.y.value, other.y.units)
                    return self.x.value == other_x and self.y.value == other_y
                else:
                    return self.x.value == other[0] \
                        and self.y.value == other[1]
            else:
                return False
        except AttributeError:
            return False

    def __add__(self, other):
        try:
            if isinstance(other, Coordinate2D):
                x = self.x.value + self.convert_units(
                    other.x.value,    other.units)
                y = self.y.value + self.convert_units(
                    other.y.value, other.units)
            elif isinstance(other, Coordinate1D):
                x = self.x.value + self.convert_units(other.value, other.units)
                y = self.y.value + self.convert_units(other.value, other.units)
            elif len(other) == len(self) and not isinstance(other, str):
                x = self.x + other[0]
                y = self.y + other[1]
            elif isinstance(other, (int, float)):
                x = self.x.value + other
                y = self.y.value + other
            else:
                raise TypeError('Can only add to items of types Coordinate2D, '
                                'Coordinate1D, int, or float.')
        except AttributeError:
            raise TypeError('Array must be of length {}'.format(len(self)))

        return Coordinate2D(x, y, self.units)

    def __mul__(self, other):
        try:
            if isinstance(other, (int, float)):
                x = self.x * other
                y = self.y * other
            elif len(other) == len(self) and not isinstance(
                    other, (str, Coordinate)):
                x = self.x * other[0]
                y = self.y * other[1]
            else:
                raise TypeError('Coordinates may only be multiplied by '
                                'numbers or arrays.')
            return Coordinate2D(x, y, self.units)
        except:
            raise TypeError('Coordinates may only be multiplied by numbers or'
                            ' arrays.')

    __rmul__ = __mul__

    def __truediv__(self, other):
        try:
            if isinstance(other, (int, float)):
                x = self.x.value / other
                y = self.y.value / other
            elif len(other) == len(self) and not isinstance(
                    other, (str, Coordinate)):
                x = self.x.value / other[0]
                y = self.y.value / other[1]
            else:
                raise TypeError('Coordinates may only be divided by numbers or'
                                ' arrays.')
        except:
            raise TypeError('Coordinates may only be divided by numbers or'
                            ' arrays.')

        return Coordinate2D(x, y, 'nm')

    def __floordiv__(self, other):
        if isinstance(other, int):
            x = self.x.value // other
            y = self.y.value // other
            return Coordinate2D(x, y, 'nm')
        else:
            return self.__truediv__(other)
