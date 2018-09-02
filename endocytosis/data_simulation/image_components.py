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
import abc
import h5py
import pickle
import weakref
from os.path import abspath
from copy import copy
from scipy.optimize import minimize
from collections import Callable

import endocytosis.contrib.gohlke.psf as psf
from fiji_tools.helpers.data_structures import TrackedList
from fiji_tools.helpers.coordinate import Coordinate
from fiji_tools.helpers.decorators import methdispatch

__all__ = [PSFFactory, FieldOfView, Cell]

# unfinished stuff
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


class PSFModelGaussian2D(PSFModelParent):
    """
    TODO: incorporate EMCCCD noise into model
    """
    model_specific_attributes = ['sigma', 'A', 'x', 'y',
                                 'mx', 'my', 'b']
    class_name = u'Gaussian2D'
    objective_function = cygauss2d.objective
    model_function = cygauss2d.model

    def __init__(self):
        super().__init__()
        self.save_upon_exit = True

    @classmethod
    def render(cls, shape, A, x, y, sigma, b, mx, my):
        """
        Returns a 2d np.ndarray containing a 2d scalar field (image) of the
        model.

        coordinate: Coordinate2D
        Location of the spot center within the output image. (0,0)
        is the top left corner.

        size: tuple
        2-member tuple containing two Coordinate1D objects of the desired
        output image dimensions.

        mdh: MetaData
        If left as None, assume pixel size is 100nm.
        """
        dx, dy = shape
        X, Y = np.mgrid[:dx, :dy]
        raveled_im = cls.model_function(A, sigma, x, y, mx, my, b, X, Y)
        return raveled_im.reshape(X.shape, dtype=np.float32)

    @classmethod
    def fit_model(cls, mdh, data=None):
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

        pixelsize: Coordinate
        If left as None, assume pixel size is 100nm.
        """
        px = mdh['pixelsize']['x']
        py = mdh['pixelsize']['y']

        if data is None:
            data = np.array(cls.data, dtype=float, copy=True)
        elif isinstance(data, np.ndarray) and data.ndim == 3:
            cls.data = np.array(data, dtype=np.uint16, copy=True)
            data = np.array(data, dtype=float)
        elif isinstance(data, (list, tuple)):
            data = np.stack(data, axis=0)
            cls.data = np.array(data, dtype=np.uint16, copy=True)
        else:
            raise('No data.')

        sigma = px / 3.
        x, y = np.array(data.shape[1:]) / 2 * np.array([px, py])
        mx, my, b = 0.01, 0.01, 0.0
        X, Y = np.mgrid[:data.shape[1], :data.shape[2]]
        X, Y = X * px, Y * py

        result = np.array([minimize(
                                cls.objective_function,
                                x0=(1., sigma, x, y, mx, my, b),
                                args=(d, X, Y)) for d in data.astype(float)])

        # TODO: set average parameter values to private class variables
        # TODO: test average parameters

    @classmethod
    def bic_function(cls, n, k, sigma):
        return n * np.log(sigma) + k * np.log(n)

    @classmethod
    def BIC(cls, data, A, x, y, sigma, b, mx, my):
        n = sigma.size
        k = len(cls.model_specific_attributes)
        sigma = (np.sum((data - cls.model_function(
            data.shape, A, x, y, sigma, b, mx, my))**2)) / n
        return bic_function(n, k, sigma)

    def bic(self, *args):
        pass

    def fit(self, data, mdh):
        px = mdh['pixelsize']['x']
        py = mdh['pixelsize']['y']

        if isinstance(data, np.ndarray) and data.ndim == 3:
            pass
        elif isinstance(data, (list, tuple)):
            data = np.stack(data, axis=0)
        else:
            raise TypeError('data must be list, tuple, or numpy.ndarray')
        sigma = px / 3.
        x, y = np.array(data.shape[1:]) / 2 * np.array([px, py])
        mx, my, b = 0.01, 0.01, 0.0
        X, Y = np.mgrid[:data.shape[1], :data.shape[2]]
        X, Y = X * px, Y * py

        # should sigma be fixed?
        result = np.array([minimize(
                                self.objective_function,
                                x0=(1., sigma, x, y, mx, my, b),
                                args=(d, X, Y)) for d in data.astype(float)])

        # TODO: set average parameter values to private class variables
        # TODO: test average parameters


def PSFFactory(psftype, dims=(4., 4.), ex_wavelen=None, em_wavelen=None,
               num_aperture=1.2, refr_index=1.333, magnification=1.0,
               underfilling=1.0, pinhole_radius=None, pinhole_shape='round',
               expsf=None, empsf=None, name=None):
    """
    Factory function for generating PSF objects of arbitrary shapes. See
    documentation for psf.PSF.
    """
    return lambda sh: psf.PSF(psftype, sh, dims, ex_wavelen, em_wavelen,
                              num_aperture, refr_index, magnification,
                              underfilling, pinhole_radius, pinhole_shape, 
                              expsf, empsf, name)


class ImageComponent(object):
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
        self._prev_render = None

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

    @abc.abstractmethod
    def render(self, *args):
        pass


class Spot(ImageComponent):
    def __init__(self, psf_factory, coordinate, parent=None):
        super().__init__(coordinate, parent)
        self._psf_factory = psf_factory
        self._prev_shape = None
        self._prev_dims = None

    def __str__(self):
        return 'Spot @ {}nm.'.format(", ".join(self.coordinate.nm))

    def render(self):
        psf = self._psf_factory(shape=self.shape)
        return self.coordinate, psf.volume()

    @property
    def shape(self):
        return tuple(np.rint(i / j, dtype=int) for i, j in zip(
                        self._psf_factory.dims, self.pixelsize['um']))


class _Cell(ImageComponent):
    def __str__(self):
        return 'Cell @ {}nm.'.format(", ".join(self.coordinate.nm))

    def render(self):
        pass


class NoiseModel(ImageComponent):
    def render(self, order, view):
        pass


class GaussianSpot2D(Spot):
    _model_cls = 'GaussianSpot2D'

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
        size = Coordinate(X[-1, -1] - X[0, 0], X[-1, -1] - X[0, 0],
                          self.coordinate.units)
        # coordinate, size, pixelsize=None
        self._prev_render = self.model.render(coord, size, self.pixelsize)
        return self._prev_render


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
