# -*- coding: utf-8 -*-
"""
Methods for conversion between shapely multipolygons and binary masks

Created on Wed Jul 12 17:16:51 2017

@author: elena
"""

# imports
from rasterio import features
from rasterio.features import shapes
from shapely.geometry import shape, MultiPolygon

# conversion from shapely multipolygon to binary mask
def multipolygon2mask(multipolygon, rows, cols, trans=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0), default_val,fill_val=0, 
                   all_touched=False, dtype=None):
    mask = features.rasterize(
            [multipolygon],
            default_value = default_val,
            fill = fill_val,
            out_shape = (rows, cols),
            transform = trans, 
            dtype = dtype)
            
    return mask
    
# conversion frombinary mask to shapely multipolygon
def mask2multipolygon(mask_data, mask, trans=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0), conn=4):    
    geom_results = (
        {'properties': {'raster_val': v}, 'geometry': s}
        for i, (s, v) 
            in enumerate(shapes(mask_data, mask=mask, connectivity = conn, transform=trans)))
    geometries = list(geom_results)    

    multi = MultiPolygon([shape(geometries[i]['geometry']) for i in range (len(geometries))])
    
    if not(multi.is_valid):        
        print('Not a valid polygon!')
        
    return multi    
        