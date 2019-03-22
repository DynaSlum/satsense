Satsense
========

|Build Status| |Codacy Badge| |Maintainability| |Test Coverage|
|Documentation Status| |DOI|

Satsense is an open source Python library for patch based land-use and
land-cover classification, initially developed for a project on deprived
neighborhood detection. However, many of the algorithms made available
through Satsense can be applied in other domains, such as ecology and
climate science.

Satsense is based on readily available open source libraries, such as
opencv for machine learning and the rasterio/gdal and netcdf libraries
for data access. It has a modular design that makes it easy to add your
own hand-crafted feature or use deep learning instead.

Detection of deprived neighborhoods is a land-use classification problem
that is traditionally solved using hand crafted features like HoG,
Lacunarity, NDXI, Pantex, Texton, and SIFT, computed from very high
resolution satellite images. One of the goals of Satsense is to
facilitate assessing the performance of these features on practical
applications. To achieve this Satsense provides an easy to use open
source reference implementation for these and other features, as well as
facilities to distribute feature computation over multiple cpu’s. In the
future the library will also provide easy access to metrics for
assessing algorithm performance.

-  satsense - library for analysing satellite images, performance
   evaluation, etc.
-  notebooks - IPython notebooks for illustrating and testing the usage
   of Satsense

We are using python 3.6/3.7 and jupyter notebook for our code.

Documentation
-------------
Can be found on `readthedocs <https://satsense.readthedocs.io>`__.

Installation
------------

Please see the `installation guide on readthedocs <https://satsense.readthedocs.io/en/latest/installation.html#installation>`__.

Contributing
------------

Contributions are very welcome! Please see
`CONTRIBUTING.md <https://github.com/DynaSlum/satsense/blob/master/CONTRIBUTING.md>`__
for our contribution guidelines.

Citing Satsense
---------------

If you use Satsense for scientific research, please cite it. You can
download citation files from
`research-software.nl <https://www.research-software.nl/software/satsense>`__.

References
----------

The collection of algorithms made available trough this package is
inspired by

    J. Graesser, A. Cheriyadat, R. R. Vatsavai, V. Chandola,
    J. Long and E. Bright, "Image Based Characterization of Formal and
    Informal Neighborhoods in an Urban Landscape", in IEEE Journal of
    Selected Topics in Applied Earth Observations and Remote Sensing,
    vol. 5, no. 4, pp. 1164-1176, Aug. 2012. doi:
    10.1109/JSTARS.2012.2190383

Jordan Graesser himself also maintains `a
library <https://github.com/jgrss/spfeas>`__ with many of these
algorithms.

Test Data
~~~~~~~~~

The test data has been extracted from the Copernicus Sentinel data 2018.

.. |Build Status| image:: https://travis-ci.com/DynaSlum/satsense.svg?branch=master
   :target: https://travis-ci.com/DynaSlum/satsense
.. |Codacy Badge| image:: https://api.codacy.com/project/badge/Grade/458c8543cd304b8387b7b114218dc57c
   :target: https://www.codacy.com/app/DynaSlum/satsense?utm_source=github.com&utm_medium=referral&utm_content=DynaSlum/satsense&utm_campaign=Badge_Grade
.. |Maintainability| image:: https://api.codeclimate.com/v1/badges/ed3655f6056f89f5e107/maintainability
   :target: https://codeclimate.com/github/DynaSlum/satsense/maintainability
.. |Test Coverage| image:: https://api.codeclimate.com/v1/badges/ed3655f6056f89f5e107/test_coverage
   :target: https://codeclimate.com/github/DynaSlum/satsense/test_coverage
.. |Documentation Status| image:: https://readthedocs.org/projects/satsense/badge/?version=latest
   :target: https://satsense.readthedocs.io/en/latest/?badge=latest
.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1463015.svg
   :target: https://doi.org/10.5281/zenodo.1463015
