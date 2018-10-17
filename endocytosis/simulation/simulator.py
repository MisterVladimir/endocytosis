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
from abc import abstractmethod
import numpy as np
import h5py
import os
import random
from collections.abc import Iterator
from fijitools.helpers.decorators import methdispatch
from fijitools.helpers.coordinate import Coordinate

from .image_components import FieldOfView, Spot
from ..io import IO
from ..helpers.data_structures import TrackedSet


class CoordinateGenerator(Iterator):
    def __init__(self, imshape, pixelsize):
        self.imshape = np.array(imshape)
        self.pixelsize = pixelsize

    def __iter__(self):
        return self.__next__()

    def initialize(self):
        pass


class VanillaCoordinateGenerator(CoordinateGenerator):
    def __next__(self):
        x, y = self.imshape
        while True:
            _x = random.uniform(0, x)
            _y = random.uniform(0, y)
            px = np.array([_x, _y])
            kwargs = {'px': px, 'nm': px * self.pixelsize['nm']}
            yield Coordinate(**kwargs)


class DensityLimitGenerator(CoordinateGenerator):
    """
    Generate a list of (x, y) coordinates at a given maximum density.

    Parameters
    ------------
    density: int
    Upper limit to the density of spots. That is, at most one spot will be
    placed in every [density, density] square ROI. For example, the default
    density is such that at most one spot will be placed in every pixel.
    """
    def __init__(self, imshape, pixelsize, density=1):
        super().__init__(imshape, pixelsize)
        self.density = density
        self.initialize()

    def __next__(self):
        while True:
            dx, dy = random.random(), random.random()
            px = next(self._indices) + [dx, dy]
            px *= self.density
            kwargs = {'px': px, 'nm': px * self.pixelsize['nm']}
            yield Coordinate(**kwargs)

    def initialize(self):
        imshape = self.imshape / self.density
        X, Y = np.mgrid[:imshape[0], :imshape[1]]
        X, Y = X.ravel(), Y.ravel()
        self.upper_limit = len(X)
        self._indices = list(zip(X, Y))
        random.shuffle(self._indices)
        self._indices = (i for i in np.array(self._indices, dtype=np.float32))


class RandomSimulator(IO):
    """
    Simmulates and saves SMS-like images to HDF5 file.

    Parameters
    ------------
    density : int or None
        If left as None, generate Spot centroids along a uniform distribution.
        Otherwise density parameter sets an upper bound on the average distance
        between any two Spots. See VanillaCoordinateGenerator and
        DensityLimitGenerator above.

    args
        Parameters to set_fov method.
    """
    def __init__(self, density=None, *args):
        self._h5file = None
        self.density = density
        self.set_fov(*args)

    def cleanup(self, delete=False):
        self.h5file.close()
        if delete and self.path:
            os.remove(self.path)

    @property
    def h5file(self):
        return self._h5file

    def set_h5file(self, path):
        """
        Creates HDF5 file where simulated data is saved.
        """
        if self._h5file:
            self.cleanup()
        self._h5file = h5py.File(path)
        self.path = path

    @property
    def field_of_view(self):
        return self._fov

    @methdispatch
    def set_fov(self, *args):
        """
        Set FieldOfView object for this Simulator. All images will be
        simulated by the FieldOfView. This method can be overloaded
        using either a FieldOfView or the parameters to the FieldOfView
        constructor.

        Parameters
        ------------
        fov : FieldOfView
            Simulate images using this FieldOfView.

        shape, pixelsize, psfmodel, noise_model
            Instantiate new FieldOfView using these parameters. See
            FieldOfView documentation.
        """
        pass

    def _set_fov_params(self):
        self.imshape = self._fov.shape
        self.pixelsize = self._fov.pixelsize
        self.psf = self._fov.psf
        self.noise = self._fov.noise

    def _set_coordinate_generator(self):
        if self.density:
            self.coordinate_generator = DensityLimitGenerator(
                self.imshape, self.pixelsize, self.density)
        else:
            self.coordinate_generator = VanillaCoordinateGenerator(
                self.imshape, self.pixelsize)

    @set_fov.register(FieldOfView)
    def _set_fov(self, fov):
        self._fov = fov
        self._set_fov_params()
        self._set_coordinate_generator()

    def _set_fov_from_args(self, shape, pixelsize, psfmodel, noise_model):
        self._fov = FieldOfView(shape, pixelsize, psfmodel, noise_model)
        self._set_fov_params()
        self._set_coordinate_generator()

    @set_fov.register(tuple)
    def _set_fov(self, shape, pixelsize, psfmodel, noise_model):
        self._set_fov_from_args(shape, pixelsize, psfmodel, noise_model)

    @set_fov.register(list)
    def _set_fov(self, shape, pixelsize, psfmodel, noise_model):
        self._set_fov_from_args(shape, pixelsize, psfmodel, noise_model)

    @set_fov.register(np.ndarray)
    def _set_fov(self, shape, pixelsize, psfmodel, noise_model):
        self._set_fov_from_args(shape, pixelsize, psfmodel, noise_model)

    # @abstractmethod
    # def _render(self, *args, **kwargs):
    #     raise NotImplementedError('Implement in child classes.')

    def render(self, t):
        return self.h5file['image']['data'][t]

    def centroids(self, t):
        return self.h5file['ground_truth']['centroid'][str(t)].value

    def simulate(self, nT, n_spots, A):
        """
        nT: int
            Number of images to simulate.

        n_spots: tuple, list or np.ndarray
            (min, max) number of spots per image. The actual number of spots
            in a given image is a uniform distribution in this range.

        A: generator
            Generates the 'A' (amplitude) parameter for each Spot. It's best to
            make this an infinite generator, but at minimum, it should produce
            as many spots as there will be in the entire dataset.
        """
        if self.density and n_spots[1] > self.coordinate_generator.upper_limit:
            limit = self.coordinate_generator.upper_limit
            raise IndexError(
                'The maximum number of Spots allowed by the ' +
                'DensityLimitGenerator is {}, '.format(limit) +
                'but an upper bound of {} '.format(n_spots[1]) +
                'spots per image was passed in.')

        if not self._h5file:
            raise IOError("No HDF5 file has been created. Please use"
                          "'set_h5file' method to do so.")
        else:
            self._h5file.clear()

        self._h5file.attrs['nT'] = nT
        self._h5file.attrs['n_spots'] = n_spots

        imgrp = self._h5file.create_group('image')
        imgrp['pixelsize'] = self._fov.pixelsize['nm']
        imgrp['pixelunit'] = 'nm'
        imgrp['DimensionOrder'] = 'TXY'

        imdset = imgrp.create_dataset(
            'data', (nT, *self.imshape), np.float32)
        means = imgrp.create_dataset('mean', shape=(nT, ), dtype=np.float32)
        stdevs = imgrp.create_dataset('stdev', shape=(nT, ), dtype=np.float32)

        # load camera parameters
        cam = self._h5file.create_group('camera')
        for k, v in self._fov.camera_metadata.items():
            _ = cam.create_dataset(k, data=v)

        # generate pseudorandom Spot coördinates
        n = np.random.randint(*n_spots, size=nT)
        cords = []
        for _n in n:
            cords.append(
                [c for c, i in zip(self.coordinate_generator, range(_n))])
            self.coordinate_generator.initialize()

        grnd = self.h5file.create_group('ground_truth')
        centroid = grnd.create_group('centroid')
        for t in range(nT):
            centroid.create_dataset(str(t), dtype=np.float32,
                                    data=np.array([c['px'] for c in cords[t]]))
            # the following is hacky but it's much faster than executing
            # sequential calls to self._fov.add_spot
            self._fov.children = \
                TrackedSet([Spot(c, self._fov.psf, next(A),
                                 self._fov.spot_shape,
                                 parent=self._fov)
                            for c in cords[t]])

            imdata = self._fov.render()
            imdset[t, :, :] = imdata
            means[t] = imdata.mean()
            stdevs[t] = imdata.std()
            # Clear all spots from a FieldOfView by zero-ing out
            # the _data attribute. Not exactly kosher to peak into
            # private variables, but it's faster than methods
            # provided by FieldOfView.
            self._fov.children = TrackedSet()
            self._fov._data[:] = 0
