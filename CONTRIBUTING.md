Contributions are very welcome. Please make sure there is a github issue
associated with with every pull request. Creating an issue is also a good
way to propose new features.

# Installation for development

Please follow the installation instructions on
[readthedocs](https://satsense.readthedocs.io/en/latest/installation.html)
to get started.

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

All public functions should have [numpy style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html).
You can build the documentation locally by running
```bash
python setup.py build_sphinx
```
Use
```bash
python setup.py build_sphinx -Ea
```
to build everying from scratch. Please check that there are no warnings.
