Installation
============

Dependencies
------------
Assuming you have `conda <https://conda.io>`_ installed and have downloaded
the satsense
`environment.yml <https://github.com/DynaSlum/satsense/blob/master/environment.yml>`_
file to the current working directory, you can install
all dependencies by running:

.. code-block:: bash

   conda create --name satsense python=3
   conda activate satsense
   conda env update --file environment.yml

If you prefer to use the package manager of your OS, use it to install
the dependencies of the `GDAL <https://pypi.org/project/GDAL/>`_ and
`netCDF4 <http://unidata.github.io/netcdf4-python/>`_ Python packages.
For example, on Ubuntu Linux 18.04 and later, you can do this by running

.. code-block:: bash

   sudo apt install libgdal-dev libnetcdf-dev

Instaling from PyPI
-------------------

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

Installing for development
--------------------------

Clone the `satsense repository <https://github.com/DynaSlum/satsense>`_,
install the dependencies as described above and run

.. code-block:: bash

   pip install -e .[dev]

Please read our
`contribution guidelines <https://github.com/DynaSlum/satsense/blob/master/CONTRIBUTING.md>`_
before starting development.
