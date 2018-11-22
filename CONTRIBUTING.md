Contributions are very welcome. Please make sure there is a github issue
associated with with every pull request. Creating an issue is also a good
way to propose new features.

# Installation for development

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

Finally, to install satsense in development mode and with development
dependencies, run
```bash
pip install -e .[dev]
```

# Testing

Please add unit tests for the code you are writing (e.g. when fixing a bug, implement
a test that demonstrates the bug is fixed). You can run the unit tests locally
with the command
```python
python setup.py test
```

# Coding style

Please make sure your code is formatted according to
[PEP8](https://www.python.org/dev/peps/pep-0008/) and docstrings are written
according to [PEP257](https://www.python.org/dev/peps/pep-0257/). Publicly visible
functions should have
[numpy style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html).

Please autoformat your code with the following commands before making a pull request:
```bash
isort satsense/my_file.py
yapf -i satsense/my_file.py
```

# Documentation

You can build the documentation locally by running
```bash
python setup.py build_sphinx
```
Please check that there are no warnings.
