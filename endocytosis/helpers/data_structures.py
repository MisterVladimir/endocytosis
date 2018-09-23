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
from addict import Dict as _Dict
import sys
from ruamel import yaml
import numpy as np
import copy

from fijitools.helpers.iteration import isiterable


class Dict(_Dict):
    @classmethod
    def flatten(cls, dic):
        def _flatten(_dic, _key):
            if isinstance(_dic, dict):
                for k, v in _dic.items():
                    _flatten(v, k)
            elif isiterable(_dic):
                # list of either dicts or non-dict values
                if all([isinstance(i, dict) for i in _dic]):
                    for item in _dic:
                        _flatten(item, _key)
                else:
                    ret.update({_key: _dic})
            else:
                ret.update({_key: _dic})
        ret = {}
        _flatten(dic, None)
        return ret


class IndexedDict(dict):
    """
    Allows setting and getting keys/values by passing in the key index.

    We cannot use an integer key to set a value to None. The workaround is to
    use a key of type slice and an iterable containing None:
    >>> d = IndexedDict()
    >>> d['a'] = 0
    >>> d.iloc(slice(1), [None])
    >>> d
    {'a': None}
    """
    def _get_with_int(self, key, value):
        return self[key]

    def _get_with_slice(self, key, value):
        return [self[k] for k in key]

    def _set_with_int(self, key, value):
        self[key] = value

    def _set_with_slice(self, key, value):
        for k, v in zip(key, value):
            self[k] = v

    def iloc(self, i, value=None):
        try:
            keys = list(self.keys())[i]
        except IndexError as e:
            raise KeyError('Key must be set via self.__setitem__ before '
                           'referencing it via the .iloc() method.') from e
        else:
            method_dict = {(True, False): self._get_with_int,
                           (True, True): self._get_with_slice,
                           (False, False): self._set_with_int,
                           (False, True): self._set_with_slice}

            try:
                method = method_dict[(value is None,
                                      isiterable(keys) and isiterable(value))]
            except KeyError as e:
                raise TypeError(
                    'If key is iterable, value must also be iterable.') from e
            else:
                return method(keys, value)


class YAMLDict(Dict):
    yaml_tag = '!YAMLDict'

    @classmethod
    def dump(cls, representer, data):
        # implement subclass-specific serializing/dumping method here
        return representer.represent_mapping(cls.yaml_tag, data)

    @classmethod
    def to_yaml(cls, representer, data):
        return cls.dump(representer, data)

    @classmethod
    def load(cls, constructor, node):
        # implement subclass-specific loading method here
        constructor.flatten_mapping(node)
        return cls(constructor.construct_mapping(node, deep=True))

    @classmethod
    def from_yaml(cls, constructor, node):
        return cls.load(constructor, node)


class TrackedSet(set):
    """
    Set that keeps track of items added and removed.
    """
    def __init__(self, iterable=None):
        self.removed = set()
        self.added = set()
        if iterable:
            super().__init__(iterable)
            self._add_addable(iterable)
        else:
            super().__init__()

    def _refresh(self):
        temp = copy.copy(self.added)
        self.added.difference_update(self.removed)
        self.removed.difference_update(temp)

    def _add_removeable(self, item):
        if isiterable(item):
            self.removed.update(item)
        else:
            self.removed.add(item)
        self._refresh()

    def _add_addable(self, item):
        if isiterable(item):
            self.added.update(item)
        else:
            self.added.add(item)
        self._refresh()

    def clear(self):
        self.removed.update(self)
        super().clear()

    def remove(self, item):
        super().remove(item)
        self._add_removeable(item)

    def difference_update(self, other):
        other = set(other)
        self._add_removeable(self.intersection(other))
        super().difference_update(other)

    def discard(self, other):
        if other in self:
            self._add_removeable(other)
            super().discard(other)

    def intersection_update(self, other):
        other = set(other)
        self._add_removeable(self.difference(other))
        super().intersection_update(other)

    def symmetric_difference_update(self, other):
        other = set(other)
        self._add_removeable(self.intersection(other))
        add = other.difference(self)
        self._add_addable(add)
        super().symmetric_difference_update(other)

    def pop(self):
        ret = super().pop()
        self._add_removeable(ret)
        return ret

    def update(self, other):
        self._add_addable(other)
        super().update(other)

    def add(self, other):
        self._add_addable(other)
        super().add(other)
