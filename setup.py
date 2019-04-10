"""Satsense package."""
import re

from setuptools import find_packages, setup


def read(filename):
    with open(filename) as file:
        return file.read()


def read_authors(citation_file):
    """Read the list of authors from .cff file."""
    authors = re.findall(
        r'family-names: (.*)$\s*given-names: (.*)',
        read(citation_file),
        re.MULTILINE,
    )
    return ', '.join(' '.join(author[::-1]) for author in authors)


setup(
    name='satsense',
    use_scm_version=True,
    url='https://github.com/DynaSlum/SateliteImaging',
    license='Apache Software License',
    author=read_authors('CITATION.cff'),
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
        'affine',
        'descartes',
        'fiona',
        'netCDF4',
        'numpy',
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
    long_description=read('README.rst'),
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    platforms='any',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Development Status :: 4 - Beta',
        'Natural Language :: English',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ])
