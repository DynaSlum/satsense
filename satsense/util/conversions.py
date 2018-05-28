# -*- coding: utf-8 -*-
"""
Methods for conversion between shapely multipolygons and binary masks

Created on Wed Jul 12 17:16:51 2017

@author: elena
"""

# imports
from rasterio import features
from rasterio.features import shapes
from shapely.geometry import MultiPolygon, shape


# conversion from shapely multipolygon to binary mask
def multipolygon2mask(multipolygon,
                      rows,
                      cols,
                      default_val,
                      all_touched=False,
                      trans=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                      fill_val=0,
                      dtype=None):
    mask = features.rasterize(
        [multipolygon],
        default_value=default_val,
        fill=fill_val,
        out_shape=(rows, cols),
        transform=trans,
        all_touched=all_touched,
        dtype=dtype)

    return mask


# conversion frombinary mask to shapely multipolygon
def mask2multipolygon(mask_data,
                      mask,
                      trans=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                      conn=4):
    geom_results = ({
        'properties': {
            'raster_val': v
        },
        'geometry': s
    } for i, (s, v) in enumerate(
        shapes(mask_data, mask=mask, connectivity=conn, transform=trans)))
    geometries = list(geom_results)

    multi = MultiPolygon(
        [shape(geometries[i]['geometry']) for i in range(len(geometries))])

    if not (multi.is_valid):
        print('Not a valid polygon, using it' 's buffer!')
        multi = multi.buffer(0)

    return multi
