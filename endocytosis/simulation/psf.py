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
import os
import sys
from copy import copy
from scipy.optimize import minimize
from collections import Callable
import numbers
from itertools import count
from fijitools.helpers.data_structures import TrackedList
from fijitools.helpers.coordinate import Coordinate
from fijitools.helpers.decorators import methdispatch
from fijitools.helpers.iteration import isiterable

import endocytosis.contrib.gohlke.psf as psf
from endocytosis.simulation.psfmodel import cygauss2d
from endocytosis.contrib.PYME.Acquire.Hardware.Simulation import fakeCam
from endocytosis.helpers.config import DEFAULT as cfg


# unfinished stuff
class _PSFModelFactory(type):
    """
    Class factory for loading PSF models saved as HDF5 files. A PSF model's
    'render' method returns a numpy.ndarray whose values represent
    the image space of the PSF. If no HDF5 file is provided, we create a
    temporary file in which model data is stored. If we want to input
    model parameters, we use the make_empty_HDF5() static method to create
    appropriately-formated HDF5 files.

    hdf5 file structure is:
        [model_name][parameters][attribute_1]
                                [attribute_2]
                                 ...
                                [attribute_n]
                    [data][parameters][]
                          [image]

    Each attribute is an attribute specific to this class' PSF model class,
    e.g. Gaussian2D model would contain 'sigma', 'A', 'x', 'y' for each data
    ROI. Attributes for each class are in cls._init_data. The model name is
    cls._model_name. [data] and [source] contain the data the model was based
    on and the filename of the image (e.g. tif file) that the data came from.
    """
    def __new__(metacls, name, bases, namespace, **kwds):
        # assign class attributes from h5py file
        try:
            filename = kwds['hdf5_filename']
        except KeyError:
            folder = os.path.abspath(os.path.curdir)
            filename = os.path.join(folder, 'temp.hdf5')
            specific = namespace['model_specfic_attributes']
            common = namespace['common_attributes']
            metacls.make_empty_HDF5(filename, name, common, specific)
        else:
            metacls._test_HDF5_filename(filename)
            assert os.path.isfile(filename), \
                "{} does not exist.".format(filename)

        # XXX: why do I need the metacls argument here???
        clsname = metacls.load_h5(metacls, filename, namespace)
        return super().__new__(metacls, clsname, bases, namespace)

    # def __subclasscheck__(cls, subclass):
    #     # surely this will be useful later...
    #     # https://stackoverflow.com/questions/40764347/
    #     # python-subclasscheck-subclasshook
    #     for attr in required_attrs:
    #         if any(attr in sub.__dict__ for sub in subclass.__mro__):
    #             continue
    #         return False
    #     return True

    @staticmethod
    def _test_HDF5_filename(filename):
        ext = os.path.splitext(filename)[1]
        h5ext = ['.h5', '.hdf5', '.hf5', '.hd5']
        assert ext in h5ext, "Input HDF5 filename has extension " + \
            "{}. Please use one of the following: {}".format(
                ext, str(h5ext)[1:-1])

    @staticmethod
    def make_empty_HDF5(filename, model_name, common, specific):
        try:
            PSFModelFactory._test_HDF5_filename(filename)
            with h5py.File(filename) as f:
                params = f.create_group(model_name)
                for s in specific:
                    params.create_dataset(name=s, shape=(), dtype='f4')
                for c in common:
                    f.create_dataset(name=c, shape=(), dtype='f4')
                f.attrs['filename'] = filename
        except Exception as e:
            print("There was an error while creating the empty HDF5 file.")
            raise Exception from e

    def load_h5(cls, filnename, clsname, namespace):
        def check_attrs(attrs, keys):
            return set(attrs).difference(set(keys))

        def dset_to_numpy(dset):
            if isiterable(dset.value):
                return np.array(dset.value, copy=True)
            elif isinstance(dset.value, numbers.Integral):
                return int(dset.value)
            elif isinstance(dset.value, numbers.Real):
                return float(dset.value)
            else:
                raise TypeError("{} not a compatible type.".format(dset.value))

        def get_name(node):
            return node.name.split('/')[-1]

        h5_kwargs = {'driver': None}
        with h5py.File(filnename, 'a', **h5_kwargs) as f:
            try:
                clsgrp = f[clsname]
            except KeyError as e:
                raise NameError("Class name not set.") from e
            # make sure all model parameters are set in the HDF5 file
            missing = check_attrs(namespace['model_specific_attributes'],
                                  clsgrp.keys())
            if missing:
                raise ValueError("Parameters {} are missing from the HDF5 "
                                 "file.".format(str(missing)[1:-1]))

            for attr in namespace['model_specific_attributes']:
                dset = clsgrp[attr]
                namespace[attr] = dset_to_numpy(dset)

            try:
                datagrp = clsgrp['data']
            except KeyError:
                pass
            else:
                missing = check_attrs(namespace['model_specific_attributes'],
                                      datagrp['parameters'].keys())
                if not missing:
                    for name, dset in datagrp['parameters'].items():
                        namespace['_' + get_name(name)] = \
                            dset_to_numpy(dset_to_numpy(dset))
                    if 'indices' in datagrp.keys():
                        ind = dset_to_numpy(datagrp['indices'].value)
                        namespace['indices'] = ind
                    else:
                        namespace['indices'] = np.arange(len(dset.value))

            namespace['dirty'] = False
            namespace['hdf5_filename'] = filnename
            return clsname

    @classmethod
    def save(cls):
        # TODO: determine whether each saveable attribute is dirty
        with h5py.File(cls.hdf5_filename, 'r+') as f:
            for attr in cls.model_specific_attributes:
                f[str(cls)][attr] = getattr(cls, attr)

            for attr in cls.common_attributes:
                f[attr] = getattr(cls, attr)


class PSFModelFactory(object):
    """
    Class factory for loading PSF models saved as HDF5 files. A PSF model's
    'render' method returns a numpy.ndarray whose values represent
    the image space of the PSF. If no HDF5 file is provided, we create a
    temporary file in which model data is stored. If we want to input
    model parameters, we use the make_empty_HDF5() static method to create
    appropriately-formated HDF5 files.

    hdf5 file structure is:
        [model_name][parameters][attribute_1]
                                [attribute_2]
                                 ...
                                [attribute_n]
                    [data][parameters][]
                          [image]

    Each attribute is an attribute specific to this class' PSF model class,
    e.g. Gaussian2D model would contain 'sigma', 'A', 'x', 'y' for each data
    ROI. Attributes for each class are in cls._init_data. The model name is
    cls._model_name. [data] and [source] contain the data the model was based
    on and the filename of the image (e.g. tif file) that the data came from.
    """
    ndim = None
    model_specific_attributes = []
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
            self.check_h5_file_structure()
            self.load_h5()

        except OSError:
            self.make_empty_HDF5(abs_h5_filename)

        self.save_upon_exit = save_upon_exit
        self.dirty = False
        self.hdf5_filename = h5_filename

    def __enter__(self):
        return self

    def __exit__(self, typ, value, traceback):
        self.cleanup()

    def _check_file_ext(self, filename):
        ext = os.path.splitext(filename)[1]
        h5ext = ['.h5', '.hdf5', '.hf5', '.hd5']
        if ext not in h5ext:
            return "Input HDF5 filename has extension " + \
                "{}. Please use one of the following: {}".format(
                    ext, str(h5ext)[1:-1])

    def _check_attrs(self, attrs, keys):
        return set(attrs).difference(set(keys))

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

    def make_empty_HDF5(self, filename):
        try:
            self.h5file = h5py.File(filename, 'a')
            cls_grp = self.h5file.create_group(self.__class__.__name__)
            param_grp = cls_grp.create_group('parameters')
            for msa in self.model_specific_attributes:
                param_grp.create_dataset(name=msa, shape=(), dtype='f4')

            data_grp = cls_grp.create_group('data')
            _ = data_grp.create_group('parameters')
            # XXX: note to self: use 'require_dataset' when creating the
            # 'image' dataset in self.h5file[self.__class__.__name__]['data']
            # and any parameter datasets in
            # self.h5file[self.__class__.__name__]['data']['parameters']

            self.h5file.attrs['h5_filename'] = filename

        except Exception as e:
            # XXX: is this already called by self.__exit__?
            self.cleanup(remove_h5_file=True)
            raise Exception("There was an error while creating the "
                            "empty HDF5 file.") from e

    def check_h5_file_structure(self):
        try:
            clsgrp = self.h5file[self.__class__.__name__]
        except KeyError as e:
            self.save_upon_exit = False
            self.cleanup()
            raise NameError("Class name not a dataset in the HDF5 "
                            "file.") from e

        missing = self._check_attrs(self.model_specific_attributes,
                                    clsgrp.keys())
        if missing:
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

    def get_leaf_name(self, node):
        return node.name.split('/')[-1]

    def load_h5(self):
        clsgrp = self.h5file[self.__class__.__name__]
        for attr in self.model_specific_attributes:
            dset = clsgrp[attr]
            setattr(self, attr, self.dataset_to_numpy(dset))

        try:
            param_grp = clsgrp['data']['parameters']
        except KeyError:
            data_grp = clsgrp.require_group('data')
            _ = data_grp.create_group('parameters')
            _ = data_grp.create_group('image')
        else:
            missing = self._check_attrs(self.model_specific_attributes,
                                        param_grp.keys())
            if not missing:
                for name, dset in param_grp.items():
                    setattr(self,
                            '_' + self.get_leaf_name(name),
                            self.dataset_to_numpy(dset))
                data_grp = clsgrp['data']
                if 'indices' in data_grp.keys():
                    dset = data_grp['indices']
                    ind = self.dataset_to_numpy(dset)
                    setattr(self, 'indices', ind)
                else:
                    setattr(self, 'indices', np.arange(len(dset.value)))

    @classmethod
    def save(cls):
        # TODO: determine whether each saveable attribute is dirty
        with h5py.File(cls.hdf5_filename, 'r+') as f:
            for attr in cls.model_specific_attributes:
                f[str(cls)][attr] = getattr(cls, attr)

            for attr in cls.common_attributes:
                f[attr] = getattr(cls, attr)


class PSFModelParent(object):
    """
    Abstract base class for all PSF models.
    """
    common_attributes = ['data', 'centers', 'square_size']
    class_name = 'PSFModelParent'
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
    """
    ndim = 2
    model_specific_attributes = ['A', 'sigma', 'x', 'y',
                                 'mx', 'my', 'b']
    objective_function = cygauss2d.objective
    model_function = cygauss2d.model
    __str__ = 'PSFModelGaussian2D'

    def __init__(self):
        super().__init__()
        self.save_upon_exit = True
        self.rendered = False

    def __repr__(self):
        attrs = "".join(["\n\t" + ': '.join((a, np.round(getattr(a), 2)))
                         for a in self.model_specific_attributes])
        return "{0!s}:\n\t{1}".format(self, attrs)

    def render(self, shape):
        """
        Returns a 2d np.ndarray containing a 2d scalar field (image) of the
        model.

        Parameters
        -----------
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
        raveled_im = self.model_function(A, sigma, x, y, mx, my, b, X, Y)
        self.rendered = True
        return raveled_im.reshape(X.shape, dtype=np.float32)

    @classmethod
    def fit_model(cls, mdh, data=None, pixelsize=None):
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
        if data is None:
            data = cls.data.astype(float)
        elif isinstance(data, np.ndarray) and 1 < data.ndim < 4:
            cls.data = np.atleast_3d(data).astype(np.uint16)
            data = cls.data.astype(float)
        elif isiterable(data):
            data = np.stack(data, axis=0, dtype=float)
            cls.data = data.astype(np.unit16)
        else:
            raise Exception('No data.')

        # initial parameters (an educated guess)
        # assume pixelsize is the same in each dimension
        ps = mdh['pixelsize']['x']
        sigma = ps * 1.25
        x, y = np.array(data.shape[1:]) / 2 * ps
        mx, my, b = 0.01, 0.01, 0.0
        X, Y = np.mgrid[:data.shape[1], :data.shape[2]]
        X, Y = X * ps, Y * ps
        A = np.max(data)

        result = (minimize(cls.objective_function,
                           x0=(A, sigma, x, y, mx, my, b),
                           args=(d, X, Y))
                  for d in data)

        success = (res['success'] for res in result)
        filtered_result = filter(None, map(lambda i, b: copy(success), result))
        cls.parameter_indices = list(filter(
            None, map(lambda i, b: i if b else False, count(), copy(success))))
        params = np.concatenate([res['x'] for res in filtered_result]).reshape(
            (-1, len(cls.model_specific_attributes)))

        if params.size > 0:
            cls.A, cls.sigma, cls.x, cls.y, cls.mx, cls.my, cls.b = \
                params.mean(0)
            cls._A, cls._sigma, cls._x, cls._y, cls._mx, cls._my, cls._b = \
                params
        else:
            raise Exception("Model fitting was not successful.")

        return result, success

    @staticmethod
    def bic_function(cls, n, k, sigma):
        return n * np.log(sigma) + k * np.log(n)

    @classmethod
    def BIC(cls, data, A, sigma, x, y, mx, my, b):
        if isinstance(data, numbers.Real):
            data = cls.data[data].astype(float)
        elif isinstance(data, np.ndarray) and data.ndim == 2:
            data = data.astype(float)

        n = 2
        k = len(cls.model_specific_attributes)
        A, sigma, x, y, mx, my, b = \
            cls.A, cls.sigma, cls.x, cls.y, cls.mx, cls.my, cls.b
        sigma = (np.sum((data - cls.model_function(
            data.shape, A, sigma, x, y, mx, my, b))**2)) / n
        return cls.bic_function(n, k, sigma)


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