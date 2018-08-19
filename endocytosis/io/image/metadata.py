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
import json


UNIT_CONVERSION = {'microns': 'um', 'nanometers': 'nm',
                   'picometers': 'pm', 'µm': 'um', 'um': 'um',
                   'nm': 'nm', 'm': 'm'}


def parse_imagej(filename, data):
    """
    Converts ImageJ metadata to dictionary.
    """
    result = {'filename': filename}

    unused_keys = \
        [' BitsPerPixel', ' IsInterleaved', ' IsRGB', ' LittleEndian',
         ' PixelType', ' Series 0 Name', 'Info', 'Labels']

    info = data['Info'].split('\n')[:12]
    info = dict([item.split(' = ') for item in info])
    result.update({k.strip(): v for k, v in info.items()
                   if k not in unused_keys})
    # replace 'spacing' key with 'pixelsize'
    result.update({'pixelsize': data['spacing']})

    # replace 'microns' with 'um', 'nanometers' with 'nm', etc.
    if data['unit'] in UNIT_CONVERSION.keys():
        result['unit'] = UNIT_CONVERSION[data['unit']]

    return result


def parse_ome(filename, data):
    result = {'filename': filename}
    for not_needed_key in ('Channel', 'TiffData', 'ID', 'Type'):
        del data[not_needed_key]
    result.update(data)
    if result['PhysicalSizeX'] == result['PhysicalSizeY'] and \
            result['PhysicalSizeXUnit'] == result['PhysicalSizeYUnit']:
        result['pixelsize'] = result['PhysicalSizeX']
    for key in result:
        if key.startswith('PhysicalSize') and key.endswith('Unit'):
            result[key] = UNIT_CONVERSION[result[key]]

    return result


class MetaData(dict):
    """
    Stores metadata as a nested dictionary.

    Parameters
    ------------
    data: raw
    Metadata in the format stored by tifffile.TiffFile.
    """
    parsing_functions = {'ImageJ': parse_imagej, 'OME': parse_ome,
                         'raw': lambda x: x}

    def __init__(self, raw, typ, filename):
        try:
            self.update(self.parsing_functions[typ](filename, raw))
        except KeyError as e:
            # probably should raise a different type of exception
            print("{} is not supported.".format(typ))
            raise e
        self.raw = raw
        self.typ = typ

    def __missing__(self, key):
        self[key] = {}
        return self[key]

    def to_JSON(self, **kwargs):
        return json.dumps(self, **kwargs)

    def save(self, path, **kwargs):
        """
        Save data as JSON file.
        """
        with open(path, 'a') as f:
            json.dump(self, f, **kwargs)
