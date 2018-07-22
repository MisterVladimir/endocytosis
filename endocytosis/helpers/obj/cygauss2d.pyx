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
cimport cython


cdef double objective(double A, double sigma, double x, double y, 
                      double mx, double my, double b, 
                      double [:,::1] data,
                      double [:,::1] X, double [:,::1] Y):

    cdef Py_ssize_t imax = data.shape[0]
    cdef Py_ssize_t jmax = data.shape[1]
    cdef double ret = 0.0
    sigma = 2*sigma**2

    for i in range(imax):
        for j in range(jmax):
            ret += (data[i, j] - A*np.exp(
                -((X[i, j] - x)**2 + (Y[i, j] - y)**2)/sigma))**2
    return ret

def model(double A, double sigma, double x, double y, 
          double mx, double my, double b, int [:,::1] X, int [:,::1] Y):
    
    cdef Py_ssize_t imax = X.shape[0]
    cdef Py_ssize_t jmax = X.shape[1]
    sigma = 2*sigma**2
    ret_py = np.zeros_like(X, dtype=np.double)
    cdef double [:,::1] ret_c = ret_py

    for i in range(imax):
        for j in range(jmax):
            ret_c [:,::1] = A*np.exp(
                -((X[i, j] - x)**2 + (Y[i, j] - y)**2)/sigma)

    return ret_py
