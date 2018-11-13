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
import argparse
import os
import numpy as np
import importlib
import sys
from vladutils.coordinate import Coordinate

from endocytosis.simulation.image_components import FieldOfView
from endocytosis.simulation.psfmodel import SimpleGaussian2D
from endocytosis.simulation.noise import NoiseModel
from endocytosis.simulation.simulator import RandomSimulator


DEFAULT_CAMERA_FILENAME = os.path.abspath(os.path.join(
    os.pardir, 'data', 'camera.yaml'))
DEFAULT_CAMERA_MODEL = 'X-9309'
PIXELSIZE = Coordinate(px=(1., 1.), nm=(80., 80.))


def get_noise_model(filename=DEFAULT_CAMERA_FILENAME,
                    camera=DEFAULT_CAMERA_MODEL):
    noise_model = NoiseModel()
    noise_model.load_camera_metadata(filename, camera)
    noise_model.use_camera(True, 30, 2)
    return noise_model


def random_number_generator(distribution, *args):
    module = importlib.import_module('numpy.random')
    function = getattr(module, distribution)
    while True:
        yield function(*args)


def main(filename, shape, frames, spots, distribution, args):
    amplitude_generator = random_number_generator(distribution, *args)

    psf = SimpleGaussian2D(sigma=100., pixelsize=PIXELSIZE)
    nm = get_noise_model()
    fov = FieldOfView(shape, PIXELSIZE, psfmodel=psf, noise_model=nm)
    with RandomSimulator(1, fov) as sim:
        sim.set_h5file(filename)
        sim.simulate(frames, spots, amplitude_generator)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename", type=str,
        help="Filename to save output. Must have an HDF5 file extension, "
        "i.e. one of '.h5, .hd5, .hdf5, .hf5'.")
    parser.add_argument(
        "-t", "--frames", type=int, default=10,
        help="Number of images to simulate. Default: 10")
    parser.add_argument(
        "-d", "--distribution", type=str, default='poisson',
        help="Name of the random distribution that generates PSF's "
        "amplitude ('A') parameter. Can be any function from numpy.random. "
        "Default: poisson")
    parser.add_argument(
        "-a", "--args", type=float, nargs="+", default=[50, ],
        help="Arguments to the distribution function. "
        "Default: (50, )")
    parser.add_argument(
        "-sh", "--shape", type=int, nargs=2, default=(512, 512),
        help="Shape of each output image. Default: (512, 512)"
    )
    parser.add_argument(
        '-sp', '--spots', nargs=2, type=int, default=(10, 100),
        help="Enter a range (min, max) of spots to add per image. The number "
        "added will be a continuous distribution between min and max. "
        "Default: (10, 100)"
    )

    kwargs = {'frames': 10, 'distribution': 'poisson', 'args': (50, ),
              'shape': (512, 512), 'spots': (10, 100)}
    kwargs.update(vars(parser.parse_args()))
    sys.exit(main(**kwargs))
