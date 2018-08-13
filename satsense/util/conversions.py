# -*- coding: utf-8 -*-
"""
Methods for conversion between shapely multipolygons and binary masks.

Created on Wed Jul 12 17:16:51 2017

@author: elena
"""
from rasterio.features import IDENTITY, rasterize, shapes
from shapely.geometry import MultiPolygon, shape


def multipolygon2mask(multipolygon,
                      out_shape,
                      transform=IDENTITY,
                      all_touched=False):
    """Convert from shapely multipolygon to binary mask."""
    mask = rasterize(
        multipolygon,
        out_shape=out_shape,
        transform=transform,
        all_touched=all_touched)

    return mask.astype(bool)


def mask2multipolygon(mask_data, mask, transform=IDENTITY, connectivity=4):
    """Convert from binary mask to shapely multipolygon."""
    geom_results = ({
        'properties': {
            'raster_val': v
        },
        'geometry': s
    } for i, (s, v) in enumerate(
        shapes(
            mask_data,
            mask=mask,
            connectivity=connectivity,
            transform=transform)))
    geometries = list(geom_results)

    multi = MultiPolygon(
        [shape(geometries[i]['geometry']) for i in range(len(geometries))])

    if not multi.is_valid:
        print('Not a valid polygon, using it' 's buffer!')
        multi = multi.buffer(0)

    return multi
