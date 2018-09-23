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

import numpy
cimport cython


def objective(float A, float sigma, float x, float y, 
              float mx, float my, float b, 
              float [:,::1] data,
              int [:,::1] X, int [:,::1] Y):

    cdef Py_ssize_t imax = data.shape[0]
    cdef Py_ssize_t jmax = data.shape[1]
    cdef double ret = 0.0
    sigma = 2*sigma**2

    for i in range(imax):
        for j in range(jmax):
            ret += (data[i, j] - A*numpy.exp(
                -((X[i, j] - x)**2 + (Y[i, j] - y)**2)/sigma))**2
    return numpy.float(ret)

def model(float A, float [::1] sigma, float x, float y, 
          float mx, float my, float b, int [:, ::1] X, int [:, ::1] Y):
    
    cdef Py_ssize_t imax = X.shape[0]
    cdef Py_ssize_t jmax = X.shape[1]
    sigma[0] = 2*sigma[0]**2
    sigma[1] = 2*sigma[1]**2
    ret_py = numpy.zeros_like(X, dtype=numpy.float32)
    cdef float [:,::1] ret_c = ret_py

    for i in range(imax):
        for j in range(jmax):
            ret_c [i, j::1] = A*numpy.exp(
                -((X[i, j] - x)**2/sigma[0] +
                  (Y[i, j] - y)**2/sigma[1]))

    return ret_py
