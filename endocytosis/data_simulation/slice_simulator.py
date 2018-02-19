# -*- coding: utf-8 -*-
import numpy as np 

def draw_gaussian(A, x0, y0, s, size): 
    X, Y = np.mgrid[:size[0], :size[1]].astype(float)
    return A*np.exp(-((X-x0)**2 + (Y - y0)**2)/(2*s**2))

class Coordinate(object): 
    def __init__(self, x, y, units='um'): 
        d = {'um' : 1e3, 'nm': 1} 
        self.x = x * d[units] 
        self.y = y * d[units] 
        self.units = units 
    
    def to_pixels(self, pixelsize): 
        return np.array((self.x, self.y))/pixelsize 
    
    def __add__(self, other): 
        x = self.x + other.x 
        y = self.y + other.y 
        return Coordinate(x, y)

class SliceComponent(object): 
    """
    Location is center of the object, where (0,0) is the parent object's 
    location. 
    """
    @property 
    def location(self): 
        try: 
            return self._location
        except AttributeError: 
            return Coordinate(0, 0)
    @location.setter
    def location(self, p): 
        self._location = p 
    
    @property
    def pixelsize(self): 
        try: 
            return self.parent.pixelsize
        except AttributeError: 
            return None 

class Spot(SliceComponent): 
    def __init__(self, A, sigma, parent=None): 
        self.A = A 
        self.parent = parent
        self.sigma = sigma
        self.overlapped = False 
    
    def render(self, halfwidth):
        X, Y = np.mgrid[:np.rint(2*halfwidth[0]/self.pixelsize), 
                        :np.rint(2*halfwidth[1]/self.pixelsize)]
        A, x, y, sigma = self.A, X[-1]/2, Y[-1]/2, self.sigma
        return A*np.exp(-((X-x)**2 + (Y - y)**2)/(2*sigma**2)) 

class Yeast(SliceComponent): 
    def __init__(self, autofluorescence, parent=None): 
        self.autofluorescence = autofluorescence
        self.parent = parent
        self.spots = [] 

class Round(Yeast): 
    def __init__(self, diameter, autofluorescence, parent=None): 
        """
        Diameter in microns. 
        """
        super().__init__(autofluorescence, parent)
        self.diameter = diameter
    
    def render(self): 
       im = np.zeros((self.diameter + 1, self.diameter + 1)) + \
                                                       self.autofluorescence
       for sp in self.spots: 
           hw = sp.sigma * 2.5
           spot_image = sp.render(hw) 
           im[sp.loc]
        
    def add_spot(self, r, theta, A, sigma=250): 
        if self.diameter < 2 * (r + 2 * sigma): 
            x = r * np.cos(theta) 
            y = r * np.sin(theta) 
            spot = Spot(A, sigma, parent=self) 
            spot.location = Coordinate(x, y)
            self.spots.append(spot)

class Rod(Yeast): 
    def __init__(self, length, width, theta): 
        super().__init__(autofluorescence, parent)
    
        
class SliceSimulator(object): 
    def __init__(self, shape, pixelsize): 
        self.data = np.zeros(shape) 
        self.shape = shape 
        self.pixelsize = pixelsize
    
    def add_yeast(self) : 
        pass 