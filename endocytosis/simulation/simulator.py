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

    def _set_fov_from_args(self, shape, pixelsize, psfmodel, noise_model):
        self._fov = FieldOfView(shape, pixelsize, psfmodel, noise_model)

    @set_fov.register(tuple)
    def _set_fov(self, shape, pixelsize, psfmodel, noise_model):
        self._set_fov_from_args(shape, pixelsize, psfmodel, noise_model)

    @set_fov.register(list)
    def _set_fov(self, shape, pixelsize, psfmodel, noise_model):
        self._set_fov_from_args(shape, pixelsize, psfmodel, noise_model)

    @set_fov.register(np.ndarray)
    def _set_fov(self, shape, pixelsize, psfmodel, noise_model):
        self._set_fov_from_args(shape, pixelsize, psfmodel, noise_model)

    @abstractmethod
    def _render(self, *args, **kwargs):
        raise NotImplementedError('Implement in child classes.')

    def render(self, nT, n_spots, A):
        """
        n_spots: tuple, list or np.ndarray
        (min, max) number of spots per image.

        A: generator
        Generates A parameter for each Spot.
        """
        imdset = self.h5file.create_dataset('image',
                                            (nT, *self._fov.shape),
                                            float)
        imdset.attrs['pixelsize'] = self._fov.pixelsize['nm']
        imdset.attrs['pixelunit'] = 'nm'
        imdset.attrs['DimensionOrder'] = 'TXY'

        cam = self.h5file.create_group('camera')
        for k, v in self._fov.camera_metadata.items():
            _ = cam.create_dataset(k, data=v)

        n = np.random.randint(*n_spots, size=nT)
        crds = iter(np.random.rand(np.sum(n), 2)*self._fov.shape)
        coords = [[Coordinate(**{'nm': next(crds)}) for _ in range(_n)]
                  for _n in n]
        grnd = self.h5file.create_group('ground_truth')
        centroid = grnd.create_group('centroid')
        for t in range(nT):
            centroid.create_dataset(
                str(t), float, np.array([c['px'] for c in coords[t]]))
            self._fov.children = \
                TrackedSet((Spot(c, self._fov.psf, next(A),
                                 self._fov.spot_shape) for c in coords[t]))
            imdset[t, :, :] = self._fov.render()
            self._fov.children = TrackedSet()
