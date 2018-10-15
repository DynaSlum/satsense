## Satsense

[![Build Status](https://travis-ci.com/DynaSlum/satsense.svg?branch=master)](https://travis-ci.com/DynaSlum/satsense)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/2a8eb394c4e64228b7f8501c2fadbc51)](https://app.codacy.com/app/DynaSlum/satsense?utm_source=github.com&utm_medium=referral&utm_content=DynaSlum/satsense&utm_campaign=badger)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1463015.svg)](https://doi.org/10.5281/zenodo.1463015)

Satsense is a library for land use classification using satellite imagery.

* satsense - library for analysing satellite images, performance evaluation, etc.
* notebooks - IPython notebooks for illustrating and testing the usage of Satsense

## Installation
We are using python 3.5/3.6 and jupyter notebook for our code.

Assuming you have [conda](https://conda.io) installed and are in the
directory where you have checked out this repository:

```bash
conda create --name satsense python=3
source activate satsense
conda env update
```

To install satsense from the git repo in development mode use:
```bash
pip install -e .
```
