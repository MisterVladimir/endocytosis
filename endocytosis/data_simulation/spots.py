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
from abc import ABC, ABCMeta
import h5py
from os.path import abspath
from copy import copy
from scipy.optimize import minimize
import pickle

from endocytosis.helpers.data_structures import TrackedList
from endocytosis.helpers.coordinate import Coordinate1D, Coordinate2D


class ImageComponent(object):
    """
    coordinate property is center of the object, where (0,0) is the parent
    object's coordinate.
    """
    @property
    def coordinate(self):
        try:
            return self._coordinate
        except AttributeError:
            return None

    @coordinate.setter
    def coordinate(self, c):
        self._coordinate = c

    def global_coordinate(self):
        def get_parent_coordinate(child):
            if hasattr(child, 'parent') and child.parent is not None:
                return self.coordinate + get_parent_coordinate(child.parent)
            else:
                return Coordinate2D(0, 0)
        return get_parent_coordinate(self)

    @property
    def pixelsize(self):
        try:
            return self._pixelsize
        except AttributeError:
            return self.parent.pixelsize

    @pixelsize.setter
    def pixelsize(self, value):
        assert isinstance(value, Coordinate1D),
        "pixelsize must be of type Coordinate1D"
        self._coordinate = value

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, p):
        if p is not None and self is not in self._parent.children:
            self._parent.children.append(self)
            self._parent.dirty = True
        elif p is None:
            self._parent.children.remove(self)
            self._parent.dirty = True

        self._parent = p


class PSFModelFactory(type):
    """
    Metaclass

    hdf5 file structure is:
        [model_name][attribute_1]
                    [attribute_2]
                    ...
                    [attribute_n]
        [data]
        [source]

    Each attribute is an attribute specific to this class' PSF model class, e.g.
    Gaussian2D model would contain 'sigma', 'A', 'x', 'y' for each data ROI.
    Attributes for each class are in cls._init_data. The model name is
    cls._model_name. [data] and [source] contain the data the model was based on and the source file of the image from where the model
    """
    def __new__(metacls, name, bases, namespace, **kwds):
        # assign class attributes from h5py file
        try:
            filename = copy(kwds['filename'])
        except KeyError:
            filename = 'temp.hdf5'

        PSFModelFactory.load(filename)
        return type.__new__(metacls, name, bases, namespace)

    # def __subclasscheck__(cls, subclass):
    #     # https://stackoverflow.com/questions/40764347/
    #     # python-subclasscheck-subclasshook
    #     for attr in required_attrs:
    #         if any(attr in sub.__dict__ for sub in subclass.__mro__):
    #             continue
    #         return False
    #     return True

    @classmethod
    def load(cls, filnename):
        # TODO: security -- prevent loading malicious data?
        h5_kwargs = {'driver': None}
        with h5py.File(path, 'a', **h5_kwargs) as f:
            f.require_group(cls.class_name)
            for attr in cls.model_specific_attributes:
                f[cls.class_name].require_dataset(attr)
                cls.__dict__[attr] = f[cls.class_name][attr]
                f[cls.class_name][attr].attrs['dirty'] = 0

            for attr in cls.common_attributes:
                f.require_dataset(attr)
                cls.__dict__[attr] = f[attr]
                f[attr].attrs['dirty'] = 0

            cls.__dict__['filename'] = f.filename

    @classmethod
    def save(cls):
        # TODO: determine whether each saveable attribute is dirty
        with h5py.File(cls.filename, 'r+') as f:
            for attr in cls.model_specific_attributes:
                f[cls.class_name][attr] = cls.__dict__[attr]

            for attr in cls.common_attributes:
                f[attr] = cls.__dict__[attr]


class PSFModelParent(object):
    common_attributes = ['data', 'centers', 'square_size']
    class_name = u'PSFModelParent'
    objective_function = None
    model_function = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.save_upon_exit and not any(exc_type, exc_value, traceback):
            self.save()
        return False

    @staticmethod
    def bic_function(*args):
        return NotImplemented


class PSFModelGaussian2D(PSFModelParent):
    """
    """
    # extension for: self.A*np.exp(-((X-x)**2 + (Y - y)**2)/(2*sigma.value**2))
    model_specific_attributes = ['sigma', 'A', 'x', 'y',
                                 'mx', 'my', 'b']
    class_name = u'Gaussian2D'
    objective_function = None
    model_function = None

    def __init__(self):
        super().__init__()
        self.save_upon_exit = True

    @classmethod
    def render(cls, data, A, sigma, x, y, mx, my, b):
        """
        Returns a 2d np.ndarray containing a 2d scalar field (image) of the
        model.

        coordinate: Coordinate2D
        Location of the spot center within the output image. Coordinate2D(0,0)
        is the top left corner.

        size: tuple
        2-member tuple containing two Coordinate1D objects of the desired
        output image dimensions.

        mdh: MetaData
        If left as None, assume pixel size is 100nm.
        """
        # TODO: maybe units shouldn't be assumed to be nanometers?
        units = 'nm'
        size.units = units
        coordinate.units = units

        sigma = np.mean(cls.sigma)
        x = coordinate.x.value
        y = coordinate.y.value

        if mdh is None:
            px = 100.
            py = 100.
        else:
            px = mdh['pixelsize']['x']
            py = mdh['pixelsize']['y']

        X, Y = np.mgrid[:int(np.ceil(size.x.value / px)),
                        :int(np.ceil(size.y.value / py))]
        X = X * px
        Y = Y * py

        mx = np.mean(cls.mx)
        my = np.mean(cls.my)
        b = np.mean(cls.b)

        raveled_im = cls.model_function(x, y, sigma, 1., mx, my, b,
                                        X.ravel(), Y.ravel())
        return raveled_im.reshape(X.shape, dtype=np.float32)

    @classmethod
    def fit_model(cls, data=None, mdh=None):
        """
        Fits list of 2d nd.array objects to 2d-gaussian PSF model. If no
        pixelsize is provided, use pixels as dimensions.

        Parameters
        -----------
        data : list, tuple, 3d np.ndarray, or None
            if list, or tuple: 2d np.ndarray objects to fit.
            if np.ndarray: 3d np.ndarray, assuming the 0th dimension contains
                           each datapoint.
            if None: use 'data' class variale

        Whatever the source, data should be raw microscopy data in uint16.

        pixelsize: Coordinate1D
        If left as None, assume pixel size is 100nm.
        """
        if pixelsize:
            px = mdh['pixelsize']['x']
            py = mdh['pixelsize']['y']
        else:
            # assume 100nm pixelsize
            px = 100.
            py = 100.

        if data is None:
            data = cls.data
        elif isinstance(data, np.ndarray) and data.ndim == 3:
            pass
        elif isinstance(data, (list, tuple)):
            data = np.stack(data, axis=0)

        cls.data = np.array(data, dtype=np.uint16, copy=True)
        sigma = px / 3.
        x, y = np.array(data.shape[1:]) / 2 * np.array([px, py])
        mx, my, b = 0.01, 0.01, 0.0
        X, Y = np.mgrid[:data.shape[1], :data.shape[2]]
        X, Y = X.ravel() * px, Y.ravel() * py

        data = data.reshape((data.shape[0], -1)) / data.max()

        result = np.array([minimize(cls.objective_function,
                                    x0=np.array((1., sigma, x, y, mx, my, b)),
                                    args=(d, X, Y)) for d in data])
        # TODO: set average parameter values to private class variables
        # TODO: test average parameters, determine accuracy of fit, BIC?

    @staticmethod
    def bic_function(n, sigma, k):
        return n * np.log(sigma) + k * np.log(n)

    @classmethod
    def get_BIC(cls, data, ):
        # bic = n * np.log(sigma) + k * np.log(n)
        # sigma = 
        pass


class Spot(ImageComponent):
    def __init__(self, coordinate, parent=None):
        self.parent = parent
        if parent is not None:
            coordinate.units = self.pixelsize.units
        self.coordinate = coordinate
        self._prev_render = None
        self.rendered = False
        self.halfwidth = None

    @property
    def units(self):
        pass

    @units.setter
    def units(self, name):
        pass

    def render_inverted(self):
        self.rendered = False
        return -1*self._prev_render


class GaussianSpot2D(Spot):
    _model_cls = GaussianSpot2D

    def __init__(self, coordinate, parent=None,
                 sigma=None, A=None, model=None):
        super().__init__(coordinate, parent)
        if model is None and path is None:
            self.model = self._model_cls(sigma, A)
        elif isinstance(model, GaussianSpot2D):
            self.model = model
        else:
            raise TypeError('Must input GaussianSpot2D instance or '
                            'sigma and A to generate model.')

    def render(self, X, Y):
        """
        X, Y: np.ndarray
        X and Y are 2d arrays containing the x and y coordinates over which to
        render the spot.
        """
        self.rendered = True
        relative_coord = self.coordinate - (X[0, 0], Y[0, 0])
        size = Coordinate2D(X[-1, -1] - X[0, 0], X[-1, -1] - X[0, 0],
                            self.coordinate.units)
        # coordinate, size, pixelsize=None
        self._prev_render = self.model.render(coord, size, self.pixelsize)
        return self._prev_render


class FieldOfView(object):
    _spot_halfwidth_pixels = 15
    PSF_sigma = Coordinate1D(250., 'nm')

    def __init__(self, size, pixelsize):
        self._data = np.zeros(size, dtype=np.float32)
        self._pixelsize = pixelsize
        self.X, self.Y = np.mgrid[:size[0], :size[1]]*pixelsize
        self._max_X, self._max_Y = self.X[-1, -1], self.Y[-1, -1]
        self.dirty = False
        self.spot_halfwidth = Coordinate1D(
            self._spot_halfwidth_pixels*self.pixelsize, 'nm')
        self.children = TrackedList()

    @property
    def data(self):
        if self.dirty:
            self._data = self._render()
            self.dirty = False
        return self._data

    def _render(self):
        def edit_data(array, coord):
            x, y = coord.to_pixels(self.pixelsize)
            dx, dy = array.shape
            self._data[x:x+dx, y:y+dy] += array

        for item in self.children.removed:
            edit_data(*item.render_inverted())
        self.children.removed = []

        for item in self.children.added:
            item.units = self.spot_halfwidth.units
            if isinstance(item, Spot):
                xi = int(np.floor(max(
                        (item.coordinate.x.value - self.spot_halfwidth.value) /
                        self.pixelsize)))
                xf = int(np.ceil(min(
                        (item.coordinate.x.value + self.spot_halfwidth.value) /
                        self.pixelsize)))
                yi = int(np.floor(max(
                        (item.coordinate.y.value - self.spot_halfwidth.value) /
                        self.pixelsize)))
                yf = int(np.ceil(min(
                        (item.coordinate.y.value + self.spot_halfwidth.value) /
                        self.pixelsize)))
                X, Y = self.X[xi:xf+1, yi:yf+1], self.Y[xi:xf+1, yi:yf+1]
                self._data[xi:xf+1, yi:yf+1] += item.render(X, Y)

        self.children.added = []
