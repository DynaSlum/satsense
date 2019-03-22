Installation
============

Installing the dependencies
---------------------------
Satsense has a few dependencies that cannot be installed from PyPI:

- the dependencies of the `GDAL <https://pypi.org/project/GDAL/>`_ Python package
- the dependencies of the `netCDF4 <http://unidata.github.io/netcdf4-python/>`_ Python package

**Ubuntu Linux 18.04 and later**

To install the above mentioned dependencies, run

.. code-block:: bash

   sudo apt install libgdal-dev libnetcdf-dev

this probably also works for other Debian-based Linux distributions.

**RPM-based Linux distributions**

To install the above mentioned dependencies, run

.. code-block:: bash

   sudo yum install gdal-devel netcdf-devel


**Conda**

Assuming you have `conda <https://conda.io>`_ installed and have downloaded
the satsense
`environment.yml <https://github.com/DynaSlum/satsense/blob/master/environment.yml>`_
file to the current working directory, you can install
all dependencies by running:

.. code-block:: bash

   conda env create --file environment.yml --name satsense

or you can install just the minimal dependencies by running

.. code-block:: bash

   conda create --name satsense libgdal libnetcdf nb_conda

Make sure to activate the environment after installation:

.. code-block:: bash

   conda activate satsense


Installing Satsense from PyPI
-----------------------------

If you did not use conda to install the dependencies, you may still
want to create and activate a virtual environment for satsense, e.g. using
`venv <https://docs.python.org/3/library/venv.html>`_

.. code-block:: bash

   python3 -m venv ~/venv/satsense
   source ~/venv/satsense/bin/activate

Next, install satsense by running

.. code-block:: bash

   pip install satsense

If you are planning on using the :ref:`notebooks`, you can
install the required extra dependencies with

.. code-block:: bash

   pip install satsense[notebooks]

Installing Satsense from source for development
-----------------------------------------------

Clone the `satsense repository <https://github.com/DynaSlum/satsense>`_,
install the dependencies as described above, go to the directory where
you have checked out satsense and run

.. code-block:: bash

   pip install -e .[dev]

or

.. code-block:: bash

   pip install -e .[dev,notebooks]

if you would also like to use the :ref:`notebooks`.

Please read our
`contribution guidelines <https://github.com/DynaSlum/satsense/blob/master/CONTRIBUTING.md>`_
before starting development.
