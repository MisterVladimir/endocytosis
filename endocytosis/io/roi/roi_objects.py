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
from struct import pack
from collections import OrderedDict
import warnings
from abc import ABC, abstractmethod

from endocytosis.helpers.iteration import current_and_next
from endocytosis.helpers.coordinate import Coordinate
from endocytosis.helpers.data_structures import RoiPropsDict
from endocytosis.helpers.iteration import isiterable
from endocytosis.io.roi import (HEADER_SIZE, HEADER2_SIZE,
                                HEADER_DTYPE, HEADER2_DTYPE,
                                OPTIONS, SUBTYPE, ROI_TYPE,
                                COLOR_DTYPE, SELECT_ROI_PARAMS)


# TODO: store ROI coordinates in physical units e.g. nanometers instead
# of pixels
# TODO: latest version not tested

class BaseROI(ABC):
    roi_type = None

    def __init__(self, common, points, props, from_ImageJ=True):
        # populate properties common to all ROI into a numpy array
        # this makes it convenient to later export self as an ImageJ bytestream
        if from_ImageJ:
            # loaded with roi_read.IJZipReader, already formated
            self.select_params = common
        elif isinstance(common, dict):
            d = OrderedDict(**SELECT_ROI_PARAMS['hdr'],
                            **SELECT_ROI_PARAMS['hdr2'])
            dtype = np.dtype(dict(names=list(d.keys()),
                                  formats=list(d.values())))
            self.select_params = np.zeros(1, dtype)
            for k, v in common.items():
                if k in self.select_params.names:
                    if isiterable(v):
                        self.select_params[k] = tuple(v)
                    else:
                        self.select_params[k] = v

            for k, v in [('magic', b'Iout'), ('version', 227),
                         ('type', ROI_TYPE[self.roi_type])]:
                self.select_params[k] = v

        self.roi_props = props
        self._top_left = None
        self._sides = None
        self._pixelsize = None

    @property
    def points(self):
        return NotImplemented

    @points.setter
    def points(self, value):
        return NotImplemented

    @property
    def roi_props(self):
        return self._roi_props

    @roi_props.setter
    def roi_props(self, value):
        if isinstance(value, (str, bytes)):
            self._roi_props = RoiPropsDict(string=str(value))
        elif isinstance(value, dict):
            self._roi_props = RoiPropsDict(**value)
        else:
            raise TypeError('roi_props may only be set with a string, bytes'
                            'or dictionary.')

    @property
    def centroid(self):
        return self._top_left + self._sides / 2

    @property
    def ctz(self):
        return self.select_params[0][['c', 't', 'z']]

    @property
    def c(self):
        return self.ctz['c']

    @property
    def t(self):
        return self.ctz['t']

    @property
    def z(self):
        return self.ctz['z']

    @property
    def subpixel(self):
        sp = all(np.isclose(np.ceil(self._top_left['px']),
                            self._top_left['px']))
        if not sp:
            self.select_params['options'] = \
                self.select_params['options'] | OPTIONS['subpixel']
        else:
            self.select_params['options'] = \
                self.select_params['options'] & ~OPTIONS['subpixel']
        return sp

    @property
    def options(self):
        return self.select_params['options']

    def _set_color(self, key, value):
        if isinstance(value, dict):
            for k, v in value.items():
                self.select_params[key][k] = str(v)
        elif isiterable(value) and len(value) == 4:
            self.select_params[key] = tuple(value)

    @property
    def fill_color(self):
        # a, r, g, b format
        return self.select_params['fill_color'][0]

    @fill_color.setter
    def fill_color(self, value):
        self._set_color('fill_color', value)

    @property
    def stroke_color(self):
        # a, r, g, b format
        return self.select_params['stroke_color'][0]

    @stroke_color.setter
    def stroke_color(self, value):
        self._set_color('stroke_color', value)

    @property
    def stroke_width(self):
        if self.subpixel:
            return self.select_params['float_stroke_with']
        else:
            return self.select_params['stroke_with']

    @stroke_width.setter
    def stroke_width(self, value):
        if self.subpixel:
            self.select_params['float_stroke_with'] = float(value)
        self.select_params['stroke_with'] = np.int16(value)

    @property
    def pixelsize(self):
        try:
            return self.roi_props['pixelsize']
        except KeyError:
            return None

    @pixelsize.setter
    def pixelsize(self, c):
        # if pixelsize has already been set once, 
        # setting a new value will raise an AttributeError
        # from self._top_left and self._sides
        if not self._top_left.pixelsize == c:
            # avoids attribute error
            self._top_left.pixelsize = c
        else:
            self._sides.pixelsize = c
        self.roi_props['pixelsize'] = c

    def to_slice(self):
        x0, y0 = self._top_left['px'].astype(int)
        dx, dy = self._sides['px'].astype(int)
        return [slice(x0, x0 + dx), slice(y0, y0 + dy)]

    @staticmethod
    def _encode_name(name):
        """
        """
        name = list(map(ord, list(name)))
        return pack('>' + 'h'*len(name), *name)

    @classmethod
    @abstractmethod
    def to_IJ(cls, roi, name, image_name=''):
        """
        Organizes ROI information common to all ROI types. Child classes
        should () methods to ()

        Arguments
        -----------
        roi: BaseROI
        Any concrete child class of BaseROI.

        name: str
        Name of the ROI.

        image_name: str (optional)
        Image name to be written into roi_props.

        Returns
        -----------
        hdr: numpy.ndarray
        dtype is HEADER_DTYPE.

        hdr2: numpy.ndarray
        dtype is HEADER2_DTYPE.

        encoded_roi_name: bytes[]
        ROI name, encoded as big endian shorts.

        roi_props: endocytosis.helpers.data_structures.RoiPropsDict
        ROI properties associated with roi.

        """
        hdr = np.zeros(1, dtype=HEADER_DTYPE)
        hdr2 = np.zeros(1, dtype=HEADER2_DTYPE)

        # hdr data
        keys = list(SELECT_ROI_PARAMS['hdr'].keys())
        hdr[keys] = roi.select_params[keys]
        try:
            hdr['hdr2_offset'] = HEADER_SIZE + 4 + \
                len(roi.points)*(4+8*int(roi.subpixel))
        except NotImplementedError:
            hdr['hdr2_offset'] = HEADER_SIZE + 4

        x0, y0 = roi.top_left['px']
        x1, y1 = (roi.top_left + roi.sides)['px']
        coords = (y0, x0, y1, x1)
        if roi.subpixel:
            hdr[['x1', 'y1', 'x2', 'y2']] = coords
        hdr[['top', 'left', 'bottom', 'right']] = tuple(map(int, coords))

        # hdr2 data
        keys2 = list(SELECT_ROI_PARAMS['hdr2'].keys())
        hdr2[keys2] = roi.select_params[keys2]

        # name is stored as shorts
        encoded_name = cls._encode_name(name)
        hdr2['name_offset'] = hdr['hdr2_offset'] + HEADER2_SIZE
        hdr2['name_length'] = len(encoded_name)

        # set roi properties (text at the end of .roi file)
        roi_props = roi.roi_props.to_IJ(image_name)
        hdr2['roi_props_offset'] = hdr['hdr2_offset'] + HEADER2_SIZE + \
            len(encoded_name)
        hdr2['roi_props_length'] = len(roi_props)

        return hdr, hdr2, encoded_name, roi_props


class RectROI(BaseROI):
    """
    Parameters
    -----------
    props: string or RoiProps
    """
    roi_type = 'rectangle'
    skipped_fields = {'hdr': ['shape_roi_size', 'subtype', 'arrow_style',
                              'aspect_ratio', 'point_type', 'arrow_head_size',
                              'rounded_rect_arc_size', 'position'],
                      'hdr2': ['overlay_label_color', 'overlay_font_size',
                               'image_opacity', 'image_size',
                               'float_stroke_width']}

    def __init__(self, bounding_rect, common, points, props='',
                 from_ImageJ=True):
        super().__init__(common, points, props, from_ImageJ)
        self._set_bounding_rect(bounding_rect)

    def _encode_points(self):
        return b''

    def _set_bounding_rect(self, br):
        # ensures that origin is top left corner of the rectangle
        top_left = np.minimum(br[:2], br[2:], dtype='f4')
        bottom_right = np.maximum(br[:2], br[2:], dtype='f4')
        sides = bottom_right - top_left
        self._top_left = Coordinate(px=top_left)
        self._sides = Coordinate(px=sides)

    @property
    def top_left(self):
        return self._top_left

    @top_left.setter
    def top_left(self, value):
        assert isinstance(value, Coordinate)
        self._top_left = value

    @property
    def sides(self):
        return self._sides

    @sides.setter
    def sides(self, value):
        # change self.top_left to maintain the rectangle same center
        assert isinstance(value, Coordinate)
        self._sides = value
        self._top_left -= value

    @property
    def points(self):
        raise NotImplementedError('')

    def asarray(self, unit='px'):
        return np.concatenate([self._top_left[unit], self._sides[unit]])

    @classmethod
    def to_IJ(cls, roi, name, image_name=''):
        hdr, hdr2, name, props = \
            super(RectROI, cls).to_IJ(roi, name, image_name)
        if not roi.__class__ == cls:
            hdr[cls.skipped_fields['hdr']] = 0
            hdr2[cls.skipped_fields['hdr2']] = 0
            hdr['type'] = ROI_TYPE[cls.roi_type]

        return (hdr.tobytes() + b'\x00\x00\x00\x00' + hdr2.tobytes() +
                name + props)


class EllipseROI(RectROI):
    """
    """
    roi_type = None

    def __init__(self, bounding_rect, common, points, props='',
                 from_ImageJ=True):
        super().__init__(bounding_rect, common, points, props, from_ImageJ)
        # self.select_params['subtype'] = SUBTYPE['ellipse']
        if not self.select_params['options'] & OPTIONS['subpixel']:
            ratio = self.sides['px'] / np.roll(self.sides['px'], 1)
            # aspect ratio = minor / major length
            self.select_params['aspect_ratio'] = np.min(ratio)
            self.roi_type = 'oval'
        else:
            self.roi_type = 'freehand'

        self.vertices = 72

    def _encode_points(self):
        # need to implement if self.subpixel
        if self.subpixel:
            return b''
        else:
            return b''

    @property
    def angle(self):
        return np.arctan2(*self.sides['px']) * 180.0 / np.pi

    @property
    def aspect_ratio(self):
        return self.select_params['aspect_ratio']

    @property
    def points(self):
        return ([], [])

    def _calculate_points(self):
        # untested
        beta1 = np.array([2*i*np.pi/self.vertices for
                          i in range(self.vertices)])
        major = np.hypot(*self.sides['px'])
        minor = self.aspect_ratio*major
        dx = np.sin(beta1) * major / 2.0
        dy = np.cos(beta1) * minor / 2.0
        beta2 = np.arctan2(dx, dy)
        rad = np.hypot(dx, dy)
        beta3 = beta2 + self.angle / 180.0 * np.pi
        dx2 = np.sin(beta3)*rad
        dy2 = np.cos(beta3)*rad
        points = self.centroid['px'][None, :] + np.array([dx2, dy2]).T
        if self.pixelsize:
            return [Coordinate(px=p, nm=p*self.pixelsize) for p in points]
        else:
            return [Coordinate(px=p) for p in points]


class PolygonROI(BaseROI):
    roi_type = 'polygon'

    def __init__(self, bounding_rect, common, points, props='',
                 from_ImageJ=True):
        super().__init__(common, points, props, from_ImageJ)
        if from_ImageJ:
            points = self._adjust_ImageJ_points(points, bounding_rect)
        self._set_points(points)

    def _update_bounding_rect(self):
        pts = np.array([c['px'] for c in self._points])
        top_left = pts.min(axis=0)
        sides = pts.max(axis=0) - top_left
        self._top_left = Coordinate(px=top_left)
        self._sides = Coordinate(px=sides)

    def _adjust_ImageJ_points(self, pts, bounding_rect):
        """
        If ImageJ ROI is not subpixel, its points are stored relative to the
        top left corner's coordinates. Here we adjust them to be relative
        to the image's top left corner.
        """
        if self.subpixel:
            return pts
        else:
            return pts + bounding_rect[:2][0].view('2i2')

    def _set_points(self, pts):
        def rollback():
            self._points = old_points
            self._update_bounding_rect()

        old_points = copy.copy(self._points)
        self._points = [Coordinate(**c) if isinstance(c, Coordinate)
                        else Coordinate(px=c) for c in pts]
        ps = filter(lambda x: getattr(x, 'pixelsize'), self._points)

        try:
            prev = next(ps)
        except StopIteration:
            # empty iterator -> no pixelsizes set in self._points
            pass
        else:
            # make sure all points whose pixelsize property is not None
            # have the same pixelsize
            for p in ps:
                if p == prev:
                    pass
                else:
                    rollback()
                    raise AttributeError('Not all pixelsizes in self._points '
                                         'are equal. Rolling back')
                prev = p

            try:
                # also sets pixelsizes of every element in self._points
                self.pixelsize = prev
            except AttributeError as e:
                rollback()
                raise AttributeError(
                    *e.args, message='Cannot set pixelsize from self._points.'
                    'Rolling back')

        self._update_bounding_rect()

    def _encode_points(self):
        # need to implement if self.subpixel
        if self.subpixel:
            return np.array([p['px'] for p in self.points])
        else:
            return np.array(
                [p['px'] for p in self.points] - self._top_left['px'],
                dtype=np.int16)

    @property
    def top_left(self):
        return self._top_left

    @property
    def sides(self):
        return self._sides

    @property
    def points(self):
        try:
            return self._points
        except AttributeError:
            return None

    @property
    def pixelsize(self):
        return self.roi_props['pixelsize']

    @pixelsize.setter
    def pixelsize(self, c):
        if not self._top_left.pixelsize == c:
            # avoids attribute error
            self._top_left.pixelsize = c

        if not self._sides.pixelsize == c:
            self._sides.pixelsize = c

        for p in self._points:
            if not p.pixelsize == c:
                p.pixelsize = c

        self.roi_props['pixelsize'] = c

    @property
    def spline_fit(self):
        return False


class PolyLineROI(PolygonROI):
    roi_type = 'polyline'

    @property
    def spline_fit(self):
        return bool(self.select_params['options'] & OPTIONS['spline_fit'])

    @spline_fit.setter
    def spline_fit(self, b):
        if b:
            self.select_params['options'] = \
                self.select_params['options'] | OPTIONS['spline_fit']
        else:
            self.select_params['options'] = \
                self.select_params['options'] & ~OPTIONS['spline_fit']


class FreeLineROI(PolygonROI):
    roi_type = 'freeline'


class TextROI(object):
    """
    notes to self:
    copied from the ImageJ Java code

    Rectangle r = roi.getBounds();
    int hdrSize = RoiEncoder.HEADER_SIZE;
    int size = getInt(hdrSize);
    int styleAndJustification = getInt(hdrSize+4);
    int style = styleAndJustification&255;
    int justification = (styleAndJustification>>8) & 3;
    boolean drawStringMode = (styleAndJustification&1024)!=0;
    int nameLength = getInt(hdrSize+8);
    int textLength = getInt(hdrSize+12);
    char[] name = new char[nameLength];
    char[] text = new char[textLength];
    for (int i=0; i<nameLength; i++)
        name[i] = (char)getShort(hdrSize+16+i*2);
    for (int i=0; i<textLength; i++)
        text[i] = (char)getShort(hdrSize+16+nameLength*2+i*2);
    double angle = version>=225?getFloat(hdrSize+16+nameLength*2+textLength*2):0f;
    Font font = new Font(new String(name), style, size);
    TextRoi roi2 = null;
    if (roi.subPixelResolution()) {
        Rectangle2D fb = roi.getFloatBounds();
        roi2 = new TextRoi(fb.getX(), fb.getY(), fb.getWidth(), fb.getHeight(), new String(text), font);
    } else
        roi2 = new TextRoi(r.x, r.y, r.width, r.height, new String(text), font);
    roi2.setStrokeColor(roi.getStrokeColor());
    roi2.setFillColor(roi.getFillColor());
    roi2.setName(getRoiName());
    roi2.setJustification(justification);
    roi2.setDrawStringMode(drawStringMode);
    roi2.setAngle(angle);
    """
    pass


def ROI(bounding_rect, common, points, props, typ, from_ImageJ=True):
    number_to_roi_class = {ROI_TYPE['rectangle']: RectROI,
                           ROI_TYPE['oval']: EllipseROI,
                           ROI_TYPE['polygon']: PolygonROI,
                           ROI_TYPE['freeline']: FreeLineROI,
                           ROI_TYPE['polyline']: PolyLineROI,
                           ROI_TYPE['freehand']: FreeLineROI}
    # try:
    if common['subtype'] == SUBTYPE['ellipse']:
        return EllipseROI(bounding_rect, common, points, props, from_ImageJ)
    elif common['subtype'] == SUBTYPE['text']:
        return TextROI(bounding_rect, common, points, props, from_ImageJ)
    else:
        cls_ = number_to_roi_class[typ]
        return cls_(bounding_rect, common, points, props, from_ImageJ)

    # except KeyError as e:
    #     try:
    #         warnings.warn('Using abstract base class _BaseROI. Some '
    #                       'information may be missing.')
    #         return BaseROI(common, coordinates, props, from_ImageJ)
    #     except:
    #         raise e
