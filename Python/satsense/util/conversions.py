# -*- coding: utf-8 -*-
"""
Methods for conversion between shapely multipolygons and binary masks

Created on Wed Jul 12 17:16:51 2017

@author: elena
"""

# imports
from rasterio import features

# conversion from shapely multipolygon to binary mask
def multipolygon2mask(multipolygon, rows, cols, trans, default_val,fill_val=0, 
                   all_touched=False, dtype=None):
    mask = features.rasterize(
            [multipolygon],
            default_value = default_val,
            fill = fill_val,
            out_shape = (rows, cols),
            transform = trans, 
            dtype = dtype)
            
    return mask