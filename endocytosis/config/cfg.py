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
from addict import Dict
import yaml

from endocytosis.helpers.data_structures import YAMLDict


DEFAULT = YAMLDict()

# for setting up datasets
# ROI attributes
DEFAULT.DATA.ROI.ROI_SIDE_LENGTH = 11
DEFAULT.DATA.ROI.ROI_ATTRIBUTE_NAME = 'centroid'

# input image attributes
DEFAULT.DATA.IMAGE.CROPPED_IMAGE_SIZE = 64
DEFAULT.DATA.IMAGE.RANDOM_CROP = True
DEFAULT.DATA.IMAGE.NORMALIZE = True
DEFAULT.DATA.IMAGE.PIXEL_MAX = 1.
DEFAULT.DATA.IMAGE.PIXEL_MIN = 0.

# Normalize the targets (subtract empirical mean, divide by empirical stddev)
DEFAULT.DATA.BBOX_NORMALIZE_TARGETS = True
# Normalize the targets using "precomputed" (or made up) means and stdevs
# (BBOX_NORMALIZE_TARGETS must also be True)
DEFAULT.DATA.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True
DEFAULT.DATA.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
DEFAULT.DATA.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)

# training hyperparameters
# Initial learning rate
DEFAULT.TRAIN.LEARNING_RATE = 0.001

# Momentum
DEFAULT.TRAIN.MOMENTUM = 0.9

# Weight decay, for regularization
DEFAULT.TRAIN.WEIGHT_DECAY = 0.0005

# Factor for reducing the learning rate
DEFAULT.TRAIN.GAMMA = 0.1

# Minibatch size (number of regions of interest [ROIs])
DEFAULT.TRAIN.BATCH_SIZE = 128

# Train bounding-box regressors
DEFAULT.TRAIN.BBOX_REG = True

# Camera parameters (from spec sheet provided by manufacturer)
# Examples:
# Karatekin lab TIRF/widefield microscope, Andor serial number X-7291
# only input parameters for 17MHz, pre-amp setting 1
kcam = Dict()
kcam.electronsPerCount = 14.39
kcam.readoutNoise = 271.81
kcam.ADOffset = 400
kcam.QE = 0.87
# a guess based on some manual for iXon cameras writing they have
# "over 590" gain elements
kcam.numGainElements = 592
# not sure where to find voltage breakdown; using default from PYME
kcam.vbreakdown = 6.6
kcam.temperature = -70.
DEFAULT.CAMERA['X-7291'] = kcam

# Yale West Campus imaging core widefield microscope
ccam = Dict()
DEFAULT.CAMERA['X-9309'] = ccam


# placeholder
DEFAULT.SIMULATION = {}


def load_yaml(filename):
    result = None
    with open(filename, 'r') as f:
        result = yaml.load(f.read())
    return Dict(**DEFAULT, **result)
