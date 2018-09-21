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
from addict import Dict
import sys
from ruamel import yaml
import numpy as np
import copy

from fijitools.helpers.iteration import isiterable


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

    def flatten(self):
        def _flatten(dic):
            for k, v in dic.items():
                if isinstance(v, dict):
                    _flatten(v)
                else:
                    ret.update({k: v})
        ret = YAMLDict()
        _flatten(self)
        return ret


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
