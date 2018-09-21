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
import h5py
import weakref
import os

from fijitools.helpers.coordinate import Coordinate
from fijitools.helpers.iteration import isiterable

from ..helpers.data_structures import TrackedSet


class AbstractSimulatedItem(ABC):
    @property
    @abstractmethod
    def pixelsize(self):
        pass


class FieldOfView(AbstractSimulatedItem):
    """
    shape: 2-tuple
    Shape of image array. Image must be 2D.

    pixelsize: float
    Nanometers per pixel.

    psfmodel: psfmodel.SimpleGaussian2D or psfmodel.Gaussian2D

    noise_model: noise.NoiseModel
    """
    def __init__(self, shape, pixelsize, psfmodel, noise_model=None):
        self.shape = shape
        self._data = np.zeros(shape, dtype=np.float32)
        self.X, self.Y = np.mgrid[:shape[0], :shape[1]]
        self._max_X, self._max_Y = self.X[-1, -1], self.Y[-1, -1]
        self.pixelsize = pixelsize
        self.psf = psfmodel
        self.noise = noise_model
        self.dirty = False
        self.children = TrackedSet()

    @property
    def pixelsize(self):
        return self._pixelsize

    @pixelsize.setter
    def pixelsize(self, value):
        self._pixelsize = value

    def render(self):
        if self.dirty:
            # self._render changes self._data
            # ideally we'd set self._data to the value returned by
            # self._render() but since self._data may be large,
            # we try to avoid making copies
            self._render()
            self.dirty = False
        if self.noise:
            return self.noise.render(self._data)
        else:
            return self._data

    @property
    def noise(self):
        return self._noise

    @noise.setter
    def noise(self, model):
        self._noise = model

    def _get_slice(self, coordinate, shape):
        """
        Where in self._data to add an item.

        coordinate: Dimensions
            Location of item center.

        shape: tuple
            Shape of rendered item in pixels.
        """
        # pixel coordinate of object center
        px = np.rint(coordinate['nm'] / self.pixelsize).astype(int)
        # how to slice into self._data
        # sldata and slobj are: [[x0, y0], 
        #                        [x1, y1]]
        sldata = px[None, ...] + np.array([-shape / 2, shape / 2], int)
        slobj = np.zeros((2, 2), int)

        slobj[0] = np.where(sldata[0] < 0, -sldata[0], 0)
        slobj[1] = np.where(sldata[1] > self._data.shape,
                            self._data.shape - sldata[1], shape)

        sldata[0] = np.maximum(sldata[0], 0)
        sldata[1] = np.minimum(sldata[1], self._data.shape)

        sldata = tuple([slice(start, stop) for start, stop in sldata.T])
        slobj = tuple([slice(start, stop) for start, stop in slobj.T])

        return sldata, slobj

    def _render_children(self, child, add=True):
        # there is probably a more elegant way to do this, like with
        # dispatching
        if isiterable(child) and not isinstance(child, dict):
            self._render_iterable(child, add)
        elif isinstance(child, dict):
            # potential bug: if it's a nested dictionary
            self._render_iterable(child.values(), add)
        elif isinstance(child, Spot):
            self._render_spot(child, add)
        elif isinstance(child, Cell):
            self._render_cell(child, add)
        else:
            raise TypeError('Cannot render child.')

    def _render_cell(self, cell, add):
        print('rendering a cell')

    def _render_spot(self, spot, add):
        sldata, slspot = self._get_slice(spot.get_global_coordinate(),
                                         spot.shape)

        arr = spot.render()
        if add:
            self._data[sldata] += arr[slspot]
            spot.rendered = True
        else:
            self._data[sldata] -= arr[slspot]
            spot.rendered = False

    def _render_iterable(self, li, add=True):
        for item in li:
            self._render_children(item, add)

    def _render(self):
        self._render_children(self.children.removed, add=False)
        self.children.removed = {}

        self._render_children(self.children.added, add=True)
        self.children.added = {}

    def add_spot(self, coordinate, A, shape):
        # creates a child Spot instance
        return Spot(coordinate, self.psf, A, shape, parent=self)


class ImageComponent(AbstractSimulatedItem):
    """
    coordinate property is center of the object, where (0,0) is the parent
    object's coordinate.
    """
    def __init__(self, coordinate, parent):
        super().__init__()
        self.rendered = False
        self.previous_render = None
        self.coordinate = coordinate
        self._parent = None
        self.parent = parent

    @property
    def coordinate(self):
        return self._coordinate

    @coordinate.setter
    def coordinate(self, c):
        self._coordinate = c

    def get_global_coordinate(self):
        def get_parent_coordinate(child):
            if hasattr(child, 'parent') and child.parent is not None:
                return child.coordinate + get_parent_coordinate(child.parent)
            else:
                return Coordinate(nm=(0, 0))

        val = get_parent_coordinate(self)
        return val

    @property
    def pixelsize(self):
        try:
            # override self.coordinate's pixelsize if parent has pixelsize
            return self.parent.pixelsize
        except AttributeError:
            return self.pixelsize

    @pixelsize.setter
    def pixelsize(self, value):
        # TODO: somehow incorporate pixelsize into the self.coordinate
        if self.parent is None or not hasattr(self.parent, 'pixelsize'):
            self.pixelsize = value
        else:
            raise AssertionError("Parent must be None or not have a pixelsize "
                                 "attribute in order to set this child's "
                                 "pixelsize.")

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, p):
        # remove self from current parent's children
        if self.rendered:
            raise("Must un-render this item from {} before assigning "
                  "new parent.".format(self.parent))

        if hasattr(self._parent, 'children'):
            self._parent.children.remove(self)
            self._parent.dirty = True
        if hasattr(p, 'children') and self not in p.children:
            p.children.add(self)
            p.dirty = True

        self._parent = p

    @abstractmethod
    def _render(self):
        raise NotImplementedError('Implement in child classes.')

    def render(self):
        if self.rendered and self.previous_render:
            raise Exception('This instance has already been rendered.')
        try:
            self.previous_render = self._render()
        except Exception as e:
            print('Cound not render from the model.')
            raise Exception from e
        else:
            return self.previous_render


class Spot(ImageComponent):
    def __init__(self, coordinate, psfmodel, A, shape, noise_model=None,
                 parent=None):
        # XXX: consider setting shape to nm units
        super().__init__(coordinate, parent)
        self.psf = psfmodel
        self.A = A
        self.shape = np.array(shape)
        self.noise = noise_model

    def __str__(self):
        return 'Spot @ {}nm.'.format(", ".join(self.coordinate['nm']))

    def _render(self):
        return self.psf.render(self.A, self.shape)


# placeholder
class Cell(ImageComponent):
    def __str__(self):
        return 'Cell @ {}nm.'.format(", ".join(self.coordinate.nm))
