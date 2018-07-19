from setuptools import find_packages, setup

with open('README.md') as readme:
    README = readme.read()

setup(
    name='satsense',
    version='0.1.0',
    url='https://github.com/DynaSlum/SateliteImaging',
    license='Apache Software License',
    author='Berend Weel, Elena Ranguelova',
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    install_requires=[
        'gdal==2.1.3',
        'descartes==1.1.0',
        'fiona==1.7.8',
        'numpy==1.12.1',
        'numba',
        'opencv-contrib-python-headless',
        'shapely==1.5.17',
    ],
    extras_require={
        'test': ['pytest', 'pytest-flake8', 'pytest-cov'],
    },
    author_email='b.weel@esiencecenter.nl',
    description=('Library for multispectral remote imaging.'),
    long_description=README,
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
