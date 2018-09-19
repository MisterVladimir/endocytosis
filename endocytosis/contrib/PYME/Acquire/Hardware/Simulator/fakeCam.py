#!/usr/bin/python

##################
# fakeCam.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##################

import numpy as np

from endocytosis.contrib.PYME.Acquire.Hardware import EMCCDTheory


class NoiseMaker(object):
    def __init__(self, QE=.8, electronsPerCount=27.32, readoutNoise=109.8,
                 TrueEMGain=0, background=0., ADOffset=967, shutterOpen=True,
                 numGainElements=536, vbreakdown=6.6, temperature=-70.,
                 fast_read_approx=False, **kwargs):

        self.QE = QE
        self.ElectronsPerCount = electronsPerCount
        self.ReadoutNoise = readoutNoise
        self.TrueEMGain = TrueEMGain
        self.background = background
        self.ADOffset = ADOffset
        self.NGainElements = numGainElements
        self.vbreakdown = vbreakdown
        self.temperature = temperature
        self.shutterOpen = shutterOpen

        # approximate readout noise
        self.approximate_read_noise = fast_read_approx

        self._ar_key = None
        self._ar_cache = None

    def _read_approx(self, im_shape):
        """
        Really dirty fast approximation to readout noise by indexing into a
        random location within a pre-calculated noise matrix. Note that this
        may result in undesired correlations in the read noise.

        Parameters
        ----------
        im_shape

        Returns
        -------

        """
        nEntries = int(np.prod(im_shape))
        ar_key = (nEntries, self.ADOffset, self.ReadoutNoise,
                  self.ElectronsPerCount)

        if not self._ar_key == ar_key or self._ar_cache is None:
            self._ar_cache = self.ADOffset + \
                (self.ReadoutNoise / self.ElectronsPerCount) * \
                np.random.normal(size=2*nEntries)
            self._ar_key = ar_key

        offset = np.random.randint(0, nEntries)
        return self._ar_cache[offset:(offset+nEntries)].reshape(im_shape)

    def noisify(self, im):
        """Add noise to image using an EMCCD noise model"""

        M = EMCCDTheory.M((80. + self.TrueEMGain)/(255 + 80.),
                          self.vbreakdown, self.temperature,
                          self.NGainElements, 2.2)
        F2 = 1.0/EMCCDTheory.FSquared(M, self.NGainElements)

        if self.approximate_read_noise:
            o = self._read_approx(im.shape)
        else:
            o = self.ADOffset + \
                (self.ReadoutNoise / self.ElectronsPerCount) * \
                np.random.standard_normal(im.shape)

        if self.shutterOpen:
            o += (M/(self.ElectronsPerCount*F2)) * \
                np.random.poisson((self.QE*F2)*(im + self.background))

        return o

    def getbg(self):
        M = EMCCDTheory.M((80. + self.TrueEMGain)/(255 + 80.), self.vbreakdown,
                          self.temperature, self.NGainElements, 2.2)
        F2 = 1.0/EMCCDTheory.FSquared(M, self.NGainElements)

        return self.ADOffset + M*(int(self.shutterOpen) *
                                  (0 + self.background) *
                                  self.QE*F2) / (self.ElectronsPerCount*F2)
