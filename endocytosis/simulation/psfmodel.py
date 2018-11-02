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
import os
from scipy.optimize import minimize
import numbers
from vladutils.data_structures import TrackedList
from vladutils.coordinate import Coordinate
from vladutils.decorators import methdispatch
from vladutils.iteration import isiterable
from vladutils.contrib.gohlke import psf

from .obj import cygauss2d


# TODO: write tests! test!
def get_final_name(item):
    """
    Parameters
    ------------
    item: h5py.Dataset or h5py.Group

    Returns
    ------------
    Leaf name in the path of this function's argument.
    """
    return item.name.split('/')[-1]


class HDF5Attribute(object):
    """
    Helper class to keep track of whether a Dataset's value has changed.

    Parameters
    ------------
    value: numpy.ndarray or number.Real

    h5path: str

    dirty: bool

    """
    def __init__(self, value, h5path, dirty=False):
        self._value = value
        self._old_value = value
        self._dirty = dirty
        self._h5path = h5path

    @property
    def dirty(self):
        return not np.all(self._value == self._old_value) and self._dirty

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._dirty = True
        self._value = val

    @property
    def h5path(self):
        return self._h5path

    @property
    def name(self):
        return get_final_name(self._h5path)


class AbstractPSFModel(ABC):
    """
    Loads PSF models saved as HDF5 files. The model's
    'render' method returns a numpy.ndarray whose values represent
    the image space of the PSF. If no HDF5 file is provided, we create a
    temporary file in which model data is stored. If we want to input
    model parameters, we use the make_empty_HDF5() static method to create
    appropriately-formated HDF5 files.

    hdf5 file structure is:
        [model_name][parameter_0]
                    [parameter_1]
                     ...
                    [parameter_m]
                    [indices]
        [image]
        [spots]['image']
               ['topleft']

    Each parameter is a parameter specific to this class' PSF model,
    e.g. Gaussian2D model would contain values for amplitude ('A'),
    standard deviation ('sigma'), and center coördinate ('x', 'y').
    Parameter names are in cls.model_parameters. Model names are
    simply the class' __name__.

    The ['spots'] dataset in the example structure should store the cropped
    ROI whose dimensions are [roi index, t, x, y]. Note that each spot
    has a 'topleft' attribute that describes its xy coördinate in the image
    from which it was cropped.

    ['image'] is the image data source of ['spots'], i.e. the original image
    from where they were cropped.
    """
    ndim = None
    model_parameters = ['indices']
    objective_function = None
    model_function = None

    def __init__(self, h5_filename, save_upon_exit=True):
        # assign class attributes from h5py file
        err_string = self._check_file_ext(h5_filename)
        if err_string:
            raise NameError(err_string)

        abs_h5_filename = os.path.abspath(h5_filename)
        self.h5file = h5py.File(abs_h5_filename, 'r+')
        self.load_h5()

        self.save_upon_exit = save_upon_exit
        self.hdf5_filename = abs_h5_filename

    def __enter__(self):
        return self

    def __exit__(self, typ, value, traceback):
        self.cleanup()

    def cleanup(self, remove_h5_file=False):
        if remove_h5_file:
            try:
                self.h5file.close()
                os.remove(self.hdf5_filename)
            except (AttributeError, FileNotFoundError):
                # h5file was never created
                return None
        elif self.save_upon_exit:
            self.save()
        self.h5file.close()

    @abstractmethod
    def render(self, *args, **kwargs):
        pass

    def _check_file_ext(self, filename):
        ext = os.path.splitext(filename)[1]
        h5ext = ['.h5', '.hdf5', '.hf5', '.hd5']
        if ext not in h5ext:
            return ("Input HDF5 filename has extension " +
                    "{}. Please use one of the following: {}".format(
                        ext, str(h5ext)[1:-1]))

    def check_attrs(self, attrs, keys):
        return set(attrs).difference(set(list(keys)))

    @property
    def dirty(self):
        return [getattr(self, a) for a in self.model_parameters
                if getattr(self, a).dirty]

    def load_h5(self):
        if 'spots' in self.h5file:
            self._spots = self.h5file['spots']
        else:
            self._spots = None

        cls_grp = self.h5file[self.__class__.__name__]
        missing = self.check_attrs(self.model_parameters,
                                   cls_grp.keys())
        if not missing:
            for attr in self.model_parameters:
                dset = cls_grp[attr]
                self.__dict__[attr] = HDF5Attribute(
                    self.dataset_to_numpy(dset), dset.name)

            if 'indices' in cls_grp.keys():
                dset = cls_grp['indices']
                ind = self.dataset_to_numpy(dset)
                self.__dict__['indices'] = HDF5Attribute(ind, dset.name)
            else:
                indices = np.arange(len(dset))
                dset = cls_grp.require_dataset('indices', len(indices), int,
                                               indices)
                self.__dict__['indices'] = HDF5Attribute(indices, dset.name)
        else:
            self.save_upon_exit = False
            self.cleanup()
            raise ValueError("Parameters {} are missing from the HDF5 "
                             "file.".format(str(missing)[1:-1]))

    def dataset_to_numpy(self, dset):
        if isiterable(dset.value):
            return np.array(dset.value, copy=True)
        elif isinstance(dset.value, numbers.Integral):
            return int(dset.value)
        elif isinstance(dset.value, numbers.Real):
            return float(dset.value)
        else:
            raise TypeError("{} not a compatible type.".format(dset.value))

    @property
    def spots(self):
        return self.h5file['/spots/image']

    def set_spots(self, arr, topleft, pixelsize=None):
        """
        Images may be big so instead of loading them into memory we read their
        data from the HDF5 file (hard drive). Once the image Dataset is set,
        it can only be changed via slicing.

        Parameters
        -----------
        arr: np.ndarray
        Image data.

        attrs
        Attributes to the Dataset. This should probably be image metadata.
        """
        sp = self.h5file.create_group('spots')
        sp.create_dataset('image', arr.shape, data=arr)
        sp.create_dataset('topleft', dtype=int, data=topleft)
        if not pixelsize:
            pixelsize = self.h5file['image'].attrs['pixelsize']
        sp.attrs['pixelsize'] = pixelsize

    def save(self):
        for attr in self.dirty:
            self.h5file[attr.name].write_direct(attr.value)


class Gaussian2D(AbstractPSFModel):
    """
    """
    ndim = 2
    model_parameters = ['A', 'sigma', 'x', 'y',
                        'mx', 'my', 'b']
    objective_function = cygauss2d.objective
    model_function = cygauss2d.model

    def __init__(self, h5_filename, save_upon_exit=True):
        super().__init__(h5_filename, save_upon_exit)
        self.rendered = False

    def __repr__(self):
        attrs = "".join(["\n\t" + ': '.join((a, np.round(getattr(a), 2)))
                         for a in self.model_parameters])
        return "{0!s}:\n\t{1}".format(self, attrs)

    def __str__(self):
        return self.__class__.__name__ + " PSF Model"

    def render(self, shape, index=None):
        """
        Returns a 2d np.ndarray generated by the model.

        Parameters
        -----------
        index: int or None

        shape: tuple, list, or np.ndarray
        """
        if index:
            A, sigma, x, y, mx, my, b = \
                [getattr(self, n).value[index] for n in self.model_parameters]
        else:
            A, sigma, x, y, mx, my, b = \
                [getattr(self, n).value.mean() for n in self.model_parameters]

        dx, dy = shape
        X, Y = np.mgrid[:dx, :dy]
        self.rendered = True
        return self.model_function(A, sigma, x, y, mx, my, b, X, Y)

    def fit_model(self, index, t):
        """
        Fits image data to the model. Note that data to be fit must be 3D, so
        at least one of 'index' or 't' must be an integer.

        Parameters
        -----------
        index: int
        ROI index.

        t: int
        Time index.
        """
        def to_tuple(arg):
            if isiterable(arg) and not isinstance(arg, dict):
                return tuple(arg)
            else:
                return arg

        # as of numpy 1.15, indices may not be lists or np.ndarrays,
        # only tuples
        index, t = to_tuple(index), to_tuple(t)
        data = np.atleast_3d(self.spots[index, t]).astype(np.float32)

        # initial parameters (an educated guess)
        sigma = np.float32(self.h5file['spots'].attrs['pixelsize'])
        x, y = np.array(data.shape[-2:] / 2., dtype=np.float32)
        mx, my, b = np.array([0.01, 0.01, 0.0], dtype=np.float32)
        X, Y = np.mgrid[:data.shape[1], :data.shape[2]]
        A = np.max(data)

        result = [minimize(self.objective_function,
                           x0=(A, sigma, x, y, mx, my, b),
                           args=(d, X, Y))
                  for d in data]

        success = [res['success'] for res in result]
        indices = np.arange(len(result), dtype=int)[success]
        cls_name = self.__class__.__name__
        self.indices = HDF5Attribute(
            indices, '/{}/parameters/indices'.format(cls_name), True)

        params = np.array([res['x'] for res in result])

        if params.size > 0:
            self.A, self.sigma, self.x, self.y, self.mx, self.my, self.b = \
                [HDF5Attribute(p, '/{}/{}'.format(cls_name, a), True) for
                 p, a in zip(params, self.model_parameters)]
        else:
            raise Exception("Model fitting was not successful.")

        return result, indices

    def bic_function(self, n, k, sigma):
        return n * np.log(sigma) + k * np.log(n)

    def BIC(self, data, A, sigma, x, y, mx, my, b):
        raise NotImplementedError


class SimpleGaussian2D(object):
    """
    Parameters
    -------------
    x, y: float
    coordinates relative to the rendered image's center.
    """
    model_function = cygauss2d.model

    def __init__(self, sigma):
        self.sigma = sigma

    def render(self, A, x, y, shape):
        shape = np.array(shape, dtype=np.float32)
        A = np.float32(A)
        x, y = np.array([x, y]) + shape / 2
        zero = np.float32(0.)
        if len(self.sigma) == 1:
            sigma = np.copy(self.sigma['px'] * np.ones(2, dtype=np.float32))
        elif len(self.sigma) == 2:
            sigma = np.copy(self.sigma['px'].astype(np.float32))
        else:
            raise TypeError('sigma must be one- or two-dimensional.')

        dx, dy = np.array(shape, dtype=np.int16)
        X, Y = np.mgrid[:dx, :dy]
        return self.model_function(A, sigma, x, y,
                                   zero, zero, zero, X, Y)


# not sure what to do with this for now
# it may be useful for other models
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
