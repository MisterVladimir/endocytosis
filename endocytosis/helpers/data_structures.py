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
import copy


class TrackedList(list):
    """
    List that keeps track of items added and removed.
    """
    removed = []
    added = []

    def _append_removeable(self, item):
        self.removed.append(item)

    def __delitem__(self, index):
        self._append_removeable(self[index])
        super().__delitem__(index)

    def clear(self):
        self.removed += list(self)
        super().clear()

    def remove(self, item):
        self._append_removeable(item)
        super().remove(item)

    def pop(self): 
        self._append_removeable(self[-1])
        super().pop()

    def _append_addable(self, item):
        if np.iterable(item) and not isinstance(item, (str,)):
            self.added += list(item)
        else:
            self.added.append(item)

    def __add__(self, item):
        ret = copy.copy(self)
        ret._append_addable(item)
        super().__add__(ret)

    def __setitem__(self, key, value):
        self._append_removeable(self[key])
        self._append_addable(value)
        super().__setitem__(key, value)

    def extend(self, items):
        self._append_addable(items)
        super().extend(items)

    def append(self, item): 
        self._append_addable(item)
        super().append(item)

    def insert(self, index, item):
        self._append_addable(item) 
        super().insert(index, item)

fresultdtype=[('tIndex', '<i4'),
              ('fitResults', [('Ag', '<f4'),('x0', '<f4'),('y0', '<f4'),('sigma', '<f4'), ('bg', '<f4'),('Ar', '<f4'),('x1', '<f4'),('y1', '<f4'),('sigmag', '<f4'), ('br', '<f4')]),
              ('fitError', [('Ag', '<f4'),('x0', '<f4'),('y0', '<f4'),('sigma', '<f4'), ('bg', '<f4'),('Ar', '<f4'),('x1', '<f4'),('y1', '<f4'),('sigmag', '<f4'), ('br', '<f4')]),
              ('startParams', [('Ag', '<f4'),('x0', '<f4'),('y0', '<f4'),('sigma', '<f4'), ('bg', '<f4'),('Ar', '<f4'),('x1', '<f4'),('y1', '<f4'),('sigmag', '<f4'), ('br', '<f4')]), 
              ('subtractedBackground', [('g','<f4'),('r','<f4')]), ('nchi2', '<f4'),
              ('resultCode', '<i4'), ('slicesUsed', [('x', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),('y', [('start', '<i4'),('stop', '<i4'),('step', '<i4')])])]

ds_dtype = [('x', '>i4')]


class NestedDict(dict):
    """
    Helper class for converting fit data from h5 format into dictionary or
    numpy structured array.
    """
    dtype = np.dtype(ds_dtype)

    def __init__(self):
        super().__init__()
#        list of all field names in our dtype
        self.all_fields = np.concatenate(
            [[k] if self.dtype[k].fields is None else
                [k] + list(self.dtype[k].fields.keys())
                for k in self.dtype.fields])

    def __missing__(self, key): 
        # make sure we're adding data appropriate for this class
        if key not in self.all_fields:
            pass
        # raise KeyError(
        # '{0} not a legal key. Use keys from self.dtype.'.format(key))
        else:
            self[key] = self.__new__(type(self))
            return self[key]

    def __call__(self, name, node):
        """
        Use this method as the argument to h5py.visititems() in order to
        load data from h5py file into self. Group and Dataset names form the
        keys of self. Dataset values are leaves.
        """
        if isinstance(node, h5py.Dataset):
            if len(name.split(r'/')) == 1:
                self[name] = node.value
            # print("Dataset:", node.name, name, node)
        elif isinstance(node, h5py.Group):
            # print("Group:", node.name, name, node)
            _name = node.name.split(r'/')[-1]
            for item in node.values():
                child_name = item.name.split(r'/')[-1]
                self[_name].__call__(child_name, item)
        else:
            pass
        return None
        
    def to_struct_array(self, shape):
        """
        Return the data as a numpy structured array of type self.dtype
        """
        def func(ret, val, dtype):
            for k in dtype.fields:
                if dtype[k].fields is None:
                    ret[k] = val[k]
                else:
                    func(ret[k], val[k], ret[k].dtype)
            return ret
        # structured array that is filled and returned
        ret = np.zeros(shape, dtype=self.dtype)
        # source of data
        source = self
        dtype = self.dtype
        return func(ret, source, dtype)
