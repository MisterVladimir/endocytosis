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
import numbers
from vladutils.coordinate import Coordinate
from vladutils.iteration import isiterable
from vladutils.data_structures import TrackedSet


class FieldOfView(object):
    """
    shape : 2-tuple
        Shape of image array. Image must be 2D.

    pixelsize : float or Coordinate
        Nanometers per pixel.

    psfmodel : psfmodel.SimpleGaussian2D or psfmodel.Gaussian2D

    noise_model : noise.NoiseModel
    """
    def __init__(self, shape, pixelsize, psfmodel, noise_model=None):
        self.shape = np.array(shape, dtype=np.uint16)
        self._data = np.zeros(shape, dtype=np.float32)
        self.X, self.Y = np.mgrid[:shape[0], :shape[1]]
        self._max_X, self._max_Y = self.X[-1, -1], self.Y[-1, -1]

        self.pixelsize = pixelsize
        self.psf = psfmodel
        self.psf.pixelsize = pixelsize
        self.noise = noise_model
        self.dirty = False
        self.children = TrackedSet()

    @property
    def pixelsize(self):
        return self._pixelsize

    @pixelsize.setter
    def pixelsize(self, value):
        """

        """
        if isinstance(value, numbers.Real):
            self._pixelsize = np.array([value, value], dtype=np.float32)

        elif isinstance(value, Coordinate):
            nm = value['nm'] / value['px']
            if len(value) == 1:
                self._pixelsize = np.array([nm, nm], dtype=np.float32)
            elif len(value) == 2 and all(
                    [i in value.keys() for i in ['nm', 'px']]):
                self._pixelsize = np.array(nm, dtype=np.float32)

        elif isiterable(value) and len(value) == 2:
            self._pixelsize = np.array(value, dtype=np.float32)

        else:
            raise TypeError('pixelsize must be a number, iterable or '
                            'Coordinate.')

    def render(self):
        if self.dirty:
            # self._render changes self._data
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
        px = np.floor(coordinate['px']).astype(np.int16)
        # how to slice into self._data
        # sldata and slobj are: [[x0, y0],
        #                        [x1, y1]]
        sldata = px[None, :] + np.ceil([-shape/2, shape/2]).astype(np.int16)
        slobj = np.zeros((2, 2), dtype=np.int16)

        slobj[0] = np.where(sldata[0] < 0, -sldata[0], 0)
        slobj[1] = np.where(sldata[1] > self.shape,
                            self.shape - sldata[1], shape)

        sldata[0] = np.maximum(sldata[0], 0)
        sldata[1] = np.minimum(sldata[1], self.shape)

        sldata = tuple([slice(start, stop) for start, stop in sldata.T])
        slobj = tuple([slice(start, stop) for start, stop in slobj.T])

        return sldata, slobj

    def _render_children(self, child, add):
        # there is probably a more elegant way to do this, like with
        # Python's function dispatching
        if isiterable(child) and not isinstance(child, dict):
            self._render_iterable(child, add)
        elif isinstance(child, dict):
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
        gc = spot.get_global_coordinate()
        # slices of self._data to which the spot is added and
        # slices of the rendered spot which are added to self._data
        sldata, slspot = self._get_slice(gc, spot.shape)
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
        self.children.removed = set()

        self._render_children(self.children.added, add=True)
        self.children.added = set()

    @property
    def spot_shape(self):
        sh = self.psf.sigma['px'] * np.array([12, 12])
        return sh.astype(int)

    def add_spot(self, coordinate, A):
        # creates a child Spot instance
        coordinate.pixelsize = self.pixelsize
        return Spot(coordinate, self.psf, A, self.spot_shape, parent=self)

    def get_cropped_roi(self, xy):
        # TODO: documentation to explain this incomprehensible bit of code
        sshape = self.spot_shape
        imshape = np.array(self.shape)
        # remove any coördinates close to the edge
        mask = np.logical_and(1 + sshape / 2 < xy,
                              xy < 1 + imshape[-2:] - sshape / 2)
        xy = xy[mask.all(1), :][:, None, :]
        # start and stop indices of crop
        bounds = xy + np.ceil([-sshape // 2, sshape // 2])[None, :]
        bounds = bounds.astype(int) + 1
        im = self.render()
        data = [im[slice(*x), slice(*y)] for x, y in bounds.swapaxes(1, 2)]
        return np.dstack(data).T, bounds[:, 0, :]

    @property
    def camera_metadata(self):
        nm = self.noise
        return {'ADOffset': nm.ADOffset, 'TrueEMGain': nm.TrueEMGain,
                'readoutNoise': nm.readoutNoise,
                'electronsPerCount': nm.electronsPerCount}

    def clear(self):
        self._data[:] = 0.
        self.children = TrackedSet()

    def save(self, path):
        with h5py.File(path) as f:
            im = self.render()
            im = f.create_dataset('image', data=im)
            im.attrs['pixelsize'] = tuple(self.pixelsize)
            im.attrs['pixelunit'] = 'nm'

            cam = f.create_group('camera')
            for k, v in self.camera_metadata.items():
                _ = cam.create_dataset(k, data=v)

            # turn self.children into an ordered iterable
            children = list(self.children)
            # save cropped versions of the spots
            sp = f.create_group('spots')
            sp.attrs['pixelsize'] = tuple(self.pixelsize)
            sp.attrs['pixelunit'] = 'nm'
            centroids = np.array([c.get_global_coordinate()['px']
                                  for c in children])
            cropped, topleft = self.get_cropped_roi(centroids)
            sp.create_dataset('image', data=cropped)
            sp.create_dataset('topleft', data=topleft)

            gt = f.create_group('ground_truth')
            gt.create_dataset('centroid', dtype=np.float32, data=centroids)
            A = np.array([c.A for c in children])
            gt.create_dataset('A', dtype=np.float32, data=A)
            sigma = self.psf.sigma['px']
            gt.create_dataset('sigma', dtype=np.float32, data=sigma)


class ImageComponent(ABC):
    """
    Abstract base class for all simulated objects.

    Parameters
    -----------
    coordinate: Coordinate
    Centroid of the ImageComponent, where (0,0) is the parent object's
    coordinate.

    parent: FieldOfView, ImageComponent or None

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
        def get_coordinate(child):
            if hasattr(child, 'parent') and child.parent is not None:
                return child.coordinate + get_coordinate(child.parent)
            else:
                return Coordinate(nm=[0, 0])
        return get_coordinate(self)

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
        self.shape = np.array(shape, dtype=np.int16)
        self.noise = noise_model

    def __str__(self):
        return 'Spot @ {}nm.'.format(", ".join(self.coordinate['nm']))

    def _render(self):
        px = self.coordinate['px']
        x, y = px - np.floor(px)
        return self.psf.render(self.A, x, y, self.shape)


# placeholder
class Cell(ImageComponent):
    def __str__(self):
        return 'Cell @ {}nm.'.format(", ".join(self.coordinate.nm))
