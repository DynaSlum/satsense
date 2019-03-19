"""Satsense package."""
from setuptools import find_packages, setup

with open('README.md') as file:
    README = file.read()

setup(
    name='satsense',
    use_scm_version=True,
    url='https://github.com/DynaSlum/SateliteImaging',
    license='Apache Software License',
    author='Berend Weel, Elena Ranguelova',
    setup_requires=[
        'pytest-runner',
        'setuptools_scm',
    ],
    tests_require=[
        'hypothesis[numpy]',
        'pytest',
        'pytest-cov',
        'pytest-env',
        'pytest-flake8',
        'pytest-html',
    ],
    install_requires=[
        'descartes',
        'fiona',
        'netCDF4!=1.4.2',
        'numpy<16',
        'opencv-contrib-python-headless<3.4.3',
        'rasterio',
        'scikit-image>=0.14.2',
        'scikit-learn',
        'scipy',
        'shapely',
    ],
    extras_require={
        'dev': [
            'hypothesis[numpy]',
            'isort',
            'pycodestyle',
            'pyflakes',
            'prospector[with_pyroma]',
            'pytest',
            'pytest-cov',
            'pytest-env',
            'pytest-flake8',
            'pytest-html',
            'sphinx',
            'sphinx_rtd_theme',
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
