from __future__ import absolute_import, print_function

import io
import os

from setuptools import find_packages, setup


def read(*names, **kwargs):
    with io.open(
            os.path.join(os.path.dirname(__file__), *names),
            encoding=kwargs.get('encoding', 'utf8'),
    ) as fp:
        return fp.read()


readme = open('README.md').read()

setup(
    name='satsense',
    version='0.1.0',
    url='https://github.com/DynaSlum/SateliteImaging',
    license='Apache Software License',
    author='Berend Weel, Elena Ranguelova',
    tests_require=['pytest'],
    install_requires=[
        'gdal==2.1.3',
        'descartes==1.1.0',
        'fiona==1.7.8',
        'numpy==1.12.1',
        'opencv-python',
        'shapely==1.5.17',
    ],
    extras_require={
        'test': ['pytest', 'pytest-flake8', 'pytest-cov'],
    },
    author_email='b.weel@esiencecenter.nl',
    description=('Library for multispectral remote imaging.'),
    long_description=readme,
    packages=find_packages(),
    include_package_data=True,
    platforms='any',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ])
