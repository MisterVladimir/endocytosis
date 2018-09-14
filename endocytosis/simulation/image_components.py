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

from fijitools.helpers.data_structures import TrackedList
from fijitools.helpers.coordinate import Coordinate
from fijitools.helpers.decorators import methdispatch

import endocytosis.contrib.gohlke.psf as psf
from endocytosis.contrib.PYME.Acquire.Hardware.Simulation import fakeCam
from endocytosis.helpers.config import DEFAULT as cfg


# most of this needs to be rewritten given how much the PSF model has changed
class ImageComponent(ABC):
    """
    coordinate property is center of the object, where (0,0) is the parent
    object's coordinate.
    """
    def __init__(self, coordinate, parent=None):
        super().__init__()
        self._parent = None
        self.parent = parent
        self.coordinate = coordinate
        self.rendered = False
        self.previous_render = None

    @property
    def coordinate(self):
        return self._coordinate

    @coordinate.setter
    def coordinate(self, c):
        self._coordinate = c

    def global_coordinate(self):
        def get_parent_coordinate(child):
            if hasattr(child, 'parent') and child.parent is not None:
                return child.coordinate + get_parent_coordinate(child.parent)
            else:
                return Coordinate(nm=(0, 0))
        return get_parent_coordinate(self)

    @property
    def pixelsize(self):
        try:
            # override self.coordinate's pixelsize if parent has pixelsize
            return self.parent.pixelsize
        except AttributeError:
            return self.coordinate.pixelsize

    @pixelsize.setter
    def pixelsize(self, value):
        assert isinstance(value, psf.Dimensions), "pixelsize must be of type"
        " Dimensions"
        if self.parent is None or not hasattr(self.parent, 'pixelsize'):
            self.coordinate.pixelsize = value
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
            p.children.append(self)
            p.dirty = True

        self._parent = p

    @abstractmethod
    def render(self, *args):
        pass


class Spot(ImageComponent):
    def __init__(self, coordinate, psf_model, noise_model=None, parent=None):
        super().__init__(coordinate, parent)
        self.psf_model = psf_model
        self.noise_model = noise_model

    def __str__(self):
        return 'Spot @ {}nm.'.format(", ".join(self.coordinate.nm))

    def render(self, shape):
        if self.rendered and self.previous_render:
            raise Exception('This instance has already been rendered.')
        try:
            self.previous_render = self.psf_model.render(shape)
        except Exception as e:
            print('Cound not render from the model.')
            raise Exception from e
        else:
            return self.coordinate, self.previous_render


class _Cell(ImageComponent):
    def __str__(self):
        return 'Cell @ {}nm.'.format(", ".join(self.coordinate.nm))

    def render(self):
        pass


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


class _FieldOfView(object):
    """
    Parameters
    -----------
    dimension_order: str
    e.g. XYZCT
    """
    def __init__(self, shape, pixelsize, dimension_order,
                 psf_factory, noise_model=None):
        self.shape = shape
        self.pixelsize = pixelsize
        self.psf_factory = psf_factory
        self.noise_model = noise_model
        self._data = np.zeros(shape, dtype=np.float32)
        self.X, self.Y = np.mgrid[:shape[0], :shape[1]]
        self._max_X, self._max_Y = self.X[-1, -1], self.Y[-1, -1]
        self.dirty = False
        self.children = TrackedList()
        self.dims = {c: i for i, c in enumerate('XYZCT')
                     if c in dimension_order.capitalize()}

    @property
    def pixelsize(self):
        return self._pixelsize

    @pixelsize.setter
    def pixelsize(self, value):
        assert isinstance(value, psf.Dimensions), "pixelsize must be of type"
        " Dimensions"
        self._pixelsize = value

    @property
    def data(self):
        if self.dirty:
            # self._render changes self._data
            # ideally we'd set self._data to the value returned by
            # self._render() but since self._data may be large,
            # we try to avoid making copies
            self._render()
            self.dirty = False
        return self._data

    @property
    def noise_model(self):
        return self._noise_model

    @noise_model.setter
    def noise_model(self, model):
        self._noise_model = model

    def _get_slice(self, coordinate, shape):
        """
        Where in self._data to add an item.

        coordinate: Dimensions
            Location of item center.

        shape: tuple
            Shape of rendered item.
        """
        px = np.rint(coordinate.nm / self.pixelsize.nm)
        ret = px[None, ...] + np.array([-shape, shape + 1])
        # make sure item doesn't extend past the edges of self._data
        ret[0] = np.maximum(ret[0], 0)
        ret[1] = np.minimum(ret[1], self.data.shape)
        return [slice(i, j) for i, j in ret.T]

    @methdispatch
    def _render_children(self, child, order, view, add=True):
        pass

    @_render_children.register(Cell)
    def _(self, cell, order, view, add=True):
        pass

    @_render_children.register(Spot)
    def _(self, spot, order, view, add=True):
        sl = self._get_slice(spot.get_global_coordinate(), spot.shape)
        sl = [sl[order[0]],
              sl[order[1]],
              sl[order[2]],
              sl[order[3]],
              sl[order[4]]]

        arr = spot.render()
        if add:
            view[sl] += arr
        else:
            view[sl] -= arr

    @_render_children.register(list)
    def _(self, li, order, view, add=True):
        for item in li:
            self._render_children(item, order, view, add)

    def _render(self):
        order = [v if k in 'XYZ' else None for k, v in self.dims.items()]
        view = self._data.view(np.float32)

        removed = self.children.removed
        self._render_children(removed, order, view, add=False)
        removed = []

        added = self.children.added
        self._render_children(added, order, view, add=True)
        added = []

        if self.noise_model:
            self.noise_model.render(order, view)

        # removed_items = ListDict()
        # for item in self.children.removed:
        #     removed_items[str(item).lower().split(' ')[0]].append(item)
        #     item.rendered = False

        # added_items = ListDict()
        # for item in self.children.added:
        #     added_items[str(item).lower().split(' ')[0]].append(item)
        #     item.rendered = True

        # self._data -= self._render_cells(removed_items['cell'])
        # self._data += self._render_cells(added_items['cell'])

        # self._data -= self._render_spots(removed_items['spot'])
        # self._data += self._render_spots(added_items['spot'])


        # def render_item(_item):
        #     if isinstance(_item, Spot):
        #         sl, arr = self._render_spot(item)
        #     if isinstance(_item, Cell):
        #         sl, arr = self._render_cell(item)
        #     return sl, arr

        # def remove(_sl, _arr):
        #     self._data[_sl] -= _arr

        # def add(_sl, _arr):
        #     self._data[_sl] += _arr

        # if not self.noise_model.rendered:
        #     self._data += self._render_noise()

        # for item in self.children.removed:
        #     sl, arr = render_item(item)
        #     remove(sl, arr)
        #     try:
        #         item.rendered = False
        #         self.children.removed.remove(item)
        #     except:
        #         print("{0} could not be removed from {1}. "
        #               "Rolling back.".format(item, self))
        #         item.rendered = True
        #         add(sl, arr)

        # for item in self.children.added:
        #     sl, arr = render_item(item)
        #     add(sl, arr)
        #     try:
        #         item.rendered = True
        #         self.children.added.remove(item)
        #     except:
        #         print("{0} could not be added to {1}. "
        #               "Rolling back.".format(item, self))
        #         item.rendered = False
        #         remove(sl, arr)

    def add_spot(self, coordinate, parent):
        # creates a child Spot instance
        Spot(self.psf_factory, coordinate, parent)


def FieldOfView(shape, pixelsize, dimension_order, psf_factory,
                noise_model=None):

    def cleanup():
        for item in obj.children:
            item.parent = None

    obj = _FieldOfView(shape, pixelsize, dimension_order, psf_factory,
                       noise_model)
    return weakref.finalize(obj, cleanup)


def Cell(*args, **kwargs):
    def cleanup():
        for item in obj.children:
            item.parent = None

    obj = _Cell(*args, **kwargs)
    return weakref.finalize(obj, cleanup)
