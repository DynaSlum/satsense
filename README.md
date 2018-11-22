## Satsense

[![Build Status](https://travis-ci.com/DynaSlum/satsense.svg?branch=master)](https://travis-ci.com/DynaSlum/satsense)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/2a8eb394c4e64228b7f8501c2fadbc51)](https://app.codacy.com/app/DynaSlum/satsense?utm_source=github.com&utm_medium=referral&utm_content=DynaSlum/satsense&utm_campaign=badger)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1463015.svg)](https://doi.org/10.5281/zenodo.1463015)

Satsense is a library for land use/cover classification using satellite imagery.

* satsense - library for analysing satellite images, performance evaluation, etc.
* notebooks - IPython notebooks for illustrating and testing the usage of Satsense

We are using python 3.6/3.7 and jupyter notebook for our code.

## Installation from github

Assuming you have [conda](https://conda.io) installed and are in the
directory where you have checked out this repository, you can install
the dependencies by running:

```bash
conda create --name satsense python=3
source activate satsense
conda env update
```

If you prefer to use the package manager of your OS, use it to install
the [GDAL](https://pypi.org/project/GDAL/) and
[netCDF4](http://unidata.github.io/netcdf4-python/) dependencies. On Ubuntu
Linux 18.04 and later, you can do so by running
```bash
sudo apt install libgdal-dev libnetcdf-dev
```
When using your OS's package manager, you may still want to create and
activate a virtual environment for satsense, e.g. using 
[venv](https://docs.python.org/3/library/venv.html)
```bash
python3 -m venv ~/venv/satsense
source ~/venv/satsense/bin/activate
```

Finally, to install satsense in development mode run
```bash
pip install -e .
```

## Citing Satsense
If you use Satsense for scientific research, please cite it. You can download citation files from [research-software.nl](https://www.research-software.nl/software/satsense).

## References

The collection of algorithms made available trough this package is inspired by
> J. Graesser, A. Cheriyadat, R. R. Vatsavai, V. Chandola, J. Long and E. Bright, "Image Based Characterization of Formal and Informal Neighborhoods in an Urban Landscape," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 5, no. 4, pp. 1164-1176, Aug. 2012.
doi: 10.1109/JSTARS.2012.2190383

Jordan Graesser himself also maintains [a library](https://github.com/jgrss/spfeas) with many of these algorithms.


### Test Data
The test data has been extracted from the Copernicus Sentinel data 2018
