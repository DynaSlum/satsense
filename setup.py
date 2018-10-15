"""Satsense package."""
from setuptools import find_packages, setup

from satsense._version import __version__

with open('README.md') as readme:
    README = readme.read()

setup(
    name='satsense',
    version=__version__,
    url='https://github.com/DynaSlum/SateliteImaging',
    license='Apache Software License',
    author='Berend Weel, Elena Ranguelova',
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    install_requires=[
        'descartes',
        'fiona',
        'gdal',
        'netCDF4',
        'numba',
        'numpy',
        'opencv-contrib-python-headless<3.4.3',
        'rasterio',
        'scikit-image',
        'scikit-learn',
        'scipy',
        'shapely',
    ],
    extras_require={
        'test': [
            'pytest',
            'pytest-cov',
            'pytest-flake8',
        ],
        'dev': [
            'isort',
            'prospector[with_pyroma]',
            'yamllint',
            'yapf',
        ],
        'notebooks': [
            'jupyter',
            'matplotlib',
            'nblint',
        ],
    },
    author_email='b.weel@esiencecenter.nl',
    description=('Library for multispectral remote imaging.'),
    long_description=README,
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    platforms='any',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ])
