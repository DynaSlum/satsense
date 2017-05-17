# -*- coding: utf-8 -*-
"""

Methods for loading and visualizinf shapefiles
Created on Wed May 17 10:26:10 2017

@author: elena
"""

# imports
from descartes.patch import PolygonPatch
import fiona
from shapely.geometry import MultiPolygon, shape

# visualization
def plot_coords(ax, ob):
    x, y = ob.xy
    ax.plot(x, y, 'o', color='#999999', zorder=1)
    
def show_multipolygon(multipolygon, axis, show_coords, extent, color, alpha, title):
        
    for polygon in multipolygon:
        if show_coords:
            plot_coords(axis, polygon.exterior)
        patch = PolygonPatch(polygon, facecolor=color, edgecolor=color, alpha=alpha, zorder=2)
        axis.add_patch(patch)
    
    xmin, ymin, xmax, ymax = extent
    xrange = [xmin, xmax]
    yrange =[ymin, ymax]
    axis.set_xlim(*xrange)
   # axis.set_xticks(range(*xrange))
    axis.set_ylim(*yrange)
   # axis.set_yticks(range(*yrange))
    axis.set_aspect(1)
            
    axis.set_title(title)
    
    return axis
    
# loading    
def load_shapefile2multipolygon(shapefilename):
    fp = fiona.open(shapefilename)
    bounds = fp.bounds
    multipol = MultiPolygon([shape(pol['geometry']) for pol in fp])
    fp.close()
    
    return multipol, bounds