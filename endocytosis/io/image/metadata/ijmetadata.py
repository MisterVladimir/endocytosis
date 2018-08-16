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

# tested 
def to_dict(filename, data):
    """
    Convert IMJMetadata to dictionary.
    """
    unused_keys = \
        [' BitsPerPixel', ' IsInterleaved', ' IsRGB', ' LittleEndian',
         ' PixelType', ' Series 0 Name', 'Info', 'Labels']

    unit_conversion = {'microns': 'um', 'nanometers': 'nm',
                       'picometers': 'pm'}

    info = data['Info'].split('\n')[:12]
    info = dict([item.split(' = ') for item in info])
    ret = {'filename': filename}
    ret.update({k.strip(): v for k, v in info.items() if k not in unused_keys})
    # replace 'spacing' key with 'pixelsize'
    ret.update({'pixelsize': data['spacing']})

    # replace 'microns' with 'um', 'nanometers' with 'nm', etc.
    if data['unit'] in unit_conversion.keys():
        ret['unit'] = unit_conversion[data['unit']]

    return ret
