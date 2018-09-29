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
from ruamel import yaml
from addict import Dict


class MyYAML(yaml.YAML):
    # copied from ruamel.yaml docs
    def dump(self, data, stream=None, **kw):
        inefficient = False
        if stream is None:
            inefficient = True
            stream = yaml.compat.StringIO()
        super().dump(data, stream, **kw)
        if inefficient:
            return stream.getvalue()


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