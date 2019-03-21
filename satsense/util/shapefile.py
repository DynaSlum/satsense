# -*- coding: utf-8 -*-
"""Methods for loading and visualizion of shapefiles.

Created on Wed May 17 10:26:10 2017

@author: elena
"""

import fiona
from descartes.patch import PolygonPatch
from shapely.geometry import MultiPolygon, mapping, shape


# visualization
def plot_coords(ax, ob):
    x, y = ob.xy
    ax.plot(x, y, 'o', color='#999999', zorder=1)


def show_multipolygon(multipolygon, axis, show_coords, extent, color, alpha,
                      title):
    """Visualize multipolygon in plot."""
    for polygon in multipolygon:
        if show_coords:
            plot_coords(axis, polygon.exterior)
        patch = PolygonPatch(
            polygon, facecolor=color, edgecolor=color, alpha=alpha, zorder=2)
        axis.add_patch(patch)

    xmin, ymin, xmax, ymax = extent
    xrange = [xmin, xmax]
    yrange = [ymin, ymax]
    axis.set_xlim(*xrange)
    # axis.set_xticks(range(*xrange))
    axis.set_ylim(*yrange)
    # axis.set_yticks(range(*yrange))
    axis.set_aspect(1)

    axis.set_title(title)

    return axis


def load_shapefile2multipolygon(filename):
    """Load a shapefile as a MultiPolygon."""
    with fiona.open(filename) as file:
        multipolygon = MultiPolygon(shape(p['geometry']) for p in file)
        bounds = file.bounds

    return multipolygon, bounds


def save_multipolygon2shapefile(multipolygon, shapefilename):
    """Save a MultiPolygon to a shapefile."""
    # define the schema
    schema = {
        'geometry': 'Polygon',
        'properties': {
            'id': 'int'
        },
    }

    # write to a shapefile
    with fiona.open(shapefilename, 'w', 'ESRI Shapefile', schema) as file:
        for i, poly in enumerate(multipolygon, start=1):
            file.write({
                'geometry': mapping(poly),
                'properties': {
                    'id': i
                },
            })
