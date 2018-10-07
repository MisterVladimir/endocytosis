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
from fijitools.helpers.decorators import methdispatch
from fijitools.helpers.coordinate import Coordinate

from .image_components import FieldOfView, Spot
from ..io import IO
from ..helpers.data_structures import TrackedSet


class RandomSimulator(IO):
    def __init__(self, path):
        self.path = path
        self.h5file = h5py.File(path)

    def cleanup(self, delete=False):
        self.h5file.close()
        if delete:
            os.remove(self.path)

    @property
    def field_of_view(self):
        return self._fov

    @methdispatch
    def set_fov(self, *args, **kwargs):
        pass

    @set_fov.register(FieldOfView)
    def _set_fov(self, fov):
        self._fov = fov
        self.imshape = self._fov.shape
        self.pixelsize = self._fov.pixelsize
        self.psf = self._fov.psf
        self.noise = self._fov.noise

    def _set_fov_from_args(self, shape, pixelsize, psfmodel, noise_model):
        self._fov = FieldOfView(shape, pixelsize, psfmodel, noise_model)
        self.imshape = self._fov.shape
        self.pixelsize = self._fov.pixelsize
        self.psf = self._fov.psf
        self.noise = self._fov.noise

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

    def random_coord_generator(self):
        x, y = self.imshape
        while True:
            _x = random.uniform(0, x)
            _y = random.uniform(0, y)
            px = np.array([_x, _y])
            yield Coordinate(**{'px': px, 'nm': px*self.pixelsize['nm']})

    def render(self, t):
        return self.h5file['image'][t]

    def centroids(self, t):
        return self.h5file['ground_truth']['centroid'][str(t)].value

    def save(self, nT, n_spots, A):
        """
        nT: int
        Number of images to simulate.

        n_spots: tuple, list or np.ndarray
        (min, max) number of spots per image. The actual number of spots in a
        given image is a uniform distribution in this range.

        A: generator
        Generates A parameter for each Spot. It's best to make this an infinite
        generator, but at minimum, it should produce as many spots as there
        will be in the entire dataset.
        """
        imdset = self.h5file.create_dataset('image',
                                            (nT, *self.imshape),
                                            np.float32)
        imdset.attrs['pixelsize'] = self._fov.pixelsize['nm']
        imdset.attrs['pixelunit'] = 'nm'
        imdset.attrs['DimensionOrder'] = 'TXY'

        cam = self.h5file.create_group('camera')
        for k, v in self._fov.camera_metadata.items():
            _ = cam.create_dataset(k, data=v)

        n = np.random.randint(*n_spots, size=nT)
        cgen = self.random_coord_generator()
        cords = [[next(cgen) for _ in range(_n)] for _n in n]
        grnd = self.h5file.create_group('ground_truth')
        centroid = grnd.create_group('centroid')
        means = []
        variances = []
        for t in range(nT):
            centroid.create_dataset(str(t), dtype=np.float32,
                                    data=np.array([c['px'] for c in cords[t]]))
            # this is a hacky but it's much faster than executing multiple
            # self._fov.add_spot methods because it adds all spots at once,
            # and...
            self._fov.children = \
                TrackedSet([Spot(c, self._fov.psf, next(A),
                                 self._fov.spot_shape,
                                 parent=self._fov)
                            for c in cords[t]])

            imdata = self._fov.render()
            imdset[t, :, :] = imdata
            means.append(imdata.mean())
            variances.append(imdata.var())
            # ...clear all spots from a FieldOfView at once by zero-ing out
            # the _data of FieldOfView instead of subtracting each spot's
            # image one at a time.
            self._fov.children = TrackedSet()
            self._fov._data = np.float32(0.)
        imdset.attrs['mean'] = np.mean(means)
        imdset.attrs['std'] = np.sqrt(np.mean(variances))
