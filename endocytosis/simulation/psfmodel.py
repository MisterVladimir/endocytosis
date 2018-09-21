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
from fijitools.helpers.data_structures import TrackedList
from fijitools.helpers.coordinate import Coordinate
from fijitools.helpers.decorators import methdispatch
from fijitools.helpers.iteration import isiterable

from ..contrib.gohlke import psf
from ..simulation.obj import cygauss2d


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
        self._dirty = dirty
        self._h5path = h5path

    @property
    def dirty(self):
        return self._dirty

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
        [model_name][parameters][attribute_0]
                                [attribute_1]
                                 ...
                                [attribute_m]

                    [data][parameters][attribute_0]
                                      [attribute_1]
                                       ...
                                      [attribute_n]
                          [indices]
        [image]

    Each attribute is an attribute specific to this class' PSF model class,
    e.g. Gaussian2D model would contain values for amplitude ('A'),
    standard deviation ('sigma'), and center coördinate ('x', 'y').
    Attribute names are in cls.model_specific_attributes. Model names are
    simply the class' __name__.

    Optionally, the [data] group may contain datasets on which the above
    'parameters' were based. For example, if 'parameters' were derived from
    the Gaussian2D model, fit to diffraction-limited spots in a 2D image, we
    might want to include the parameters ('A', 'sigma', etc.) of each spot as
    array datasets. Once the dataset is loaded, ['data']'s datasets are bound
    to protected variables named after the datasets' keys.

    #########################################################################
    The ['image'] dataset in the example structure should store the cropped
    ROI as a 3d array whose dimensions are [index, x, y].
    #########################################################################
    """
    ndim = None
    model_specific_attributes = []
    data_attributes = ['indices', '']
    objective_function = None
    model_function = None

    def __init__(self, h5_filename='temp.h5', save_upon_exit=True):
        # assign class attributes from h5py file
        err_string = self._check_file_ext(h5_filename)
        if err_string:
            raise NameError(err_string)

        abs_h5_filename = os.path.abspath(h5_filename)
        try:
            self.h5file = h5py.File(abs_h5_filename, 'r+')
            has_image = self.check_h5_file_structure()
            self.load_h5(has_image)

        except OSError:
            self.make_empty_HDF5(abs_h5_filename)

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
            return "Input HDF5 filename has extension " + \
                "{}. Please use one of the following: {}".format(
                    ext, str(h5ext)[1:-1])

    def check_attrs(self, attrs, keys):
        """
        Make sure there's enough data in the HDF5 file.
        """
        return set(attrs).difference(set(keys))

    @property
    def dirty(self):
        return [getattr(self, a) for a in
                self.model_specific_attributes + self.data_attributes]

    def make_empty_HDF5(self, filename):
        try:
            self._image = None
            self.h5file = h5py.File(filename, 'a')
            cls_grp = self.h5file.create_group(self.__class__.__name__)
            param_grp = cls_grp.create_group('parameters')
            for msa in self.model_specific_attributes:
                dset = param_grp.create_dataset(name=msa, shape=(), dtype='f4')
                self.__dict__[msa] = HDF5Attribute(0., dset.name)

            data_grp = cls_grp.create_group('data')
            _ = data_grp.create_group('parameters')
            # XXX: note to self: use 'require_dataset' when creating the
            # 'image' dataset in self.h5file[self.__class__.__name__]
            # and any parameter datasets in
            # self.h5file[self.__class__.__name__]['data']['parameters']

            self.h5file.attrs['h5_filename'] = filename
            self.h5file.attrs['imshape'] = 0

        except Exception as e:
            # XXX: is this already called by self.__exit__?
            self.cleanup(remove_h5_file=True)
            raise Exception("There was an error while creating the "
                            "empty HDF5 file.") from e

    def load_h5(self, has_image):
        if has_image:
            self._image = self.h5file['image']
        else:
            self._image = None

        cls_grp = self.h5file[self.__class__.__name__]
        for attr in self.model_specific_attributes:
            dset = cls_grp[attr]
            self.__dict__[attr] = HDF5Attribute(self.dataset_to_numpy(dset),
                                                dset.name)

        try:
            param_grp = cls_grp['data']['parameters']
        except KeyError:
            data_grp = cls_grp.require_group('data')
            _ = data_grp.create_group('parameters')
        else:
            missing = self.check_attrs(self.model_specific_attributes,
                                       param_grp.keys())
            if not missing:
                for name, dset in param_grp.items():
                    self.__dict__['_' + name] = HDF5Attribute(
                        self.dataset_to_numpy(dset), dset.name)
                data_grp = cls_grp['data']
                if 'indices' in data_grp.keys():
                    dset = data_grp['indices']
                    ind = self.dataset_to_numpy(dset)
                    self.__dict__['indices'] = HDF5Attribute(ind, dset.name)
                else:
                    self.__dict__['indices'] = HDF5Attribute(
                        np.arange(len(dset.value)), dset.name)

    def check_h5_file_structure(self):
        try:
            clsgrp = self.h5file[self.__class__.__name__]
        except KeyError as e:
            self.save_upon_exit = False
            self.cleanup()
            raise NameError("Class name not a dataset in the HDF5 "
                            "file.") from e

        missing = self.check_attrs(self.model_specific_attributes,
                                   clsgrp.keys())
        if missing:
            self.save_upon_exit = False
            self.cleanup()
            raise ValueError("Parameters {} are missing from the HDF5 "
                             "file.".format(str(missing)[1:-1]))

        grp_names = [get_final_name(k) for k in self.h5file.keys()]
        if 'image' in grp_names:
            return True
        else:
            return False

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
    def image(self):
        return self._image

    def set_image(self, arr, **attrs):
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
        im = self.h5file.create_dataset('image', arr.shape, data=arr)
        for k, v in attrs.items():
            im.attrs[k] = v
        self._image = im

    def save(self):
        for attr in self.dirty:
            if isinstance(attr.value, np.ndarray):
                shape = attr.value.shape
                typ = attr.value.dtype
            else:
                shape = ()
                typ = type(attr.value)
            try:
                del self.h5file[attr.h5path]
            except KeyError:
                pass
            self.h5file.create_dataset(attr.h5path, shape, typ)
            self.h5file[attr.h5path] = attr.value


class Gaussian2D(AbstractPSFModel):
    """
    """
    ndim = 2
    model_specific_attributes = ['A', 'sigma', 'x', 'y',
                                 'mx', 'my', 'b']
    objective_function = cygauss2d.objective
    model_function = cygauss2d.model

    def __init__(self, h5_filename='temp.h5', save_upon_exit=True):
        super().__init__(h5_filename, save_upon_exit)
        self.rendered = False

    def __repr__(self):
        attrs = "".join(["\n\t" + ': '.join((a, np.round(getattr(a), 2)))
                         for a in self.model_specific_attributes])
        return "{0!s}:\n\t{1}".format(self, attrs)

    def __str__(self):
        return self.__class__.__name__ + " PSF Model"

    def render(self, shape, index=None):
        """
        Returns a 2d np.ndarray containing a 2d image of the model.

        Parameters
        -----------
        index: int or None

        shape: tuple, list, or np.ndarray

        """
        if index:
            A, sigma, x, y, mx, my, b = \
                [getattr(self, '_' + n).value[index]
                 for n in self.model_specific_attributes]
        else:
            A, sigma, x, y, mx, my, b = \
                [getattr(self, n).value
                 for n in self.model_specific_attributes]

        dx, dy = shape
        X, Y = np.mgrid[:dx, :dy]
        raveled_im = self.model_function(A, sigma, x, y, mx, my, b, X, Y)
        self.rendered = True
        return raveled_im.reshape(X.shape, dtype=np.float32)

    def fit_model(self, index=None):
        """
        Fits images stored in the /data Dataset of self.h5file to Gaussian
        PSFs.
        """
        data = np.atleast_3d(self.image).astype(float)
        if index:
            data = np.atleast_3d(data[index])

        # initial parameters (an educated guess)
        sigma = 1.25
        x, y = np.array(data.shape[1:])
        mx, my, b = 0.01, 0.01, 0.0
        X, Y = np.mgrid[:data.shape[1], :data.shape[2]]
        A = np.max(data)

        result = [minimize(self.objective_function,
                           x0=(A, sigma, x, y, mx, my, b),
                           args=(d, X, Y))
                  for d in data]

        success = [res['success'] for res in result]
        cls_name = self.__class__.__name__
        self.indices = HDF5Attribute(
            success, '/{}/data/parameters/indices'.format(cls_name), True)

        params = np.array([res['x'] for res in result])

        if params.size > 0:
            self.A, self.sigma, self.x, self.y, self.mx, self.my, self.b = \
                [HDF5Attribute(p, '/{}/{}'.format(cls_name, a), True) for
                 p, a in zip(params.mean(0), self.model_specific_attributes)]
            (self._A, self._sigma, self._x, self._y, self._mx, self._my,
             self._b) = [HDF5Attribute(
                 p, '/{}/data/parameters/_{}'.format(cls_name, a), True)
                 for p, a in zip(params.T, self.model_specific_attributes)]
        else:
            raise Exception("Model fitting was not successful.")

        return result, success

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

    def __init__(self, sigma, x, y):
        self.sigma, self.x, self.y = sigma, x, y

    def render(self, A, shape):
        dx, dy = shape
        X, Y = np.mgrid[:dx, :dy]
        x = dx // 2. + self.x
        y = dy // 2. + self.y
        return self.model_function(A, self.sigma, x, y,
                                   0, 0, 0, X, Y)


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