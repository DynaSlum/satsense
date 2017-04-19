# Satsense
Satsense is a library for remote sensing using satelite imagery.

It is based on gdal and numpy.

## Installing Satsense:
To install satsense from the git repo in development mode use:
```bash
pip install -e .
```

## Examples:
Loading a file from the worldview2 satelite and displaying it:
```python
from satsense.util import load_from_file, get_rgb_image
from satsense.features import WORLDVIEW2
import matplotlib.pyplot as plt

imagefile = '/path/to/file/WorldView.tif'
bands = WORLDVIEW2 # The ordering of the bands in a worldview2 file

# Load the file. This will give the raw gdal file as well as a numpy
# ndarray with the bands loaded (not normalized)
dataset, image = load_from_file(imagefile)

# Convert the image to an rgb image. The original image is not
# yet normalized
true_color = get_rgb_image(image, bands, normalized=False)

plt.imshow(true_color)
```

Calculating the rgNDVI of the image:
```python
from satsense.features import rgNDVI, print_ndvi_stats

ndvi = rgNDVI(image, bands=bands)

print_ndvi_stats(ndvi)

# Display the rgNDVI inverted, because with rgNDVI low values means vegetation
plt.imshow(-ndvi)
```
