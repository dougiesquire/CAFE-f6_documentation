.. CAFE-f6_skill_assessment documentation master file, created by
   sphinx-quickstart.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documentation for CAFE-f6 skill assessment
==============================================

Welcome to the documentation for my project assessing CAFE-f6. The project is organised as follows:

::

   ├── Makefile            <- Makefile with commands like `make data`
   ├── data
   │   ├── config          <- Configuration files for processing the raw data
   │   ├── processed       <- The postprocessed data
   │   ├── raw             <- The original, immutable data (or symlinks to them)
   │   └── testing         <- The data used while checking CAFE-f6 forecast reproducibility
   │
   ├── docs                <- The Sphinx documentation
   │
   ├── notebooks           <- Jupyter notebooks containing analyses. Numbers are used for
   │                          ordering where appropriate
   │
   ├── references          <- Data dictionaries, manuals, and all other explanatory materials.
   │
   ├── reports             <- Generated analysis as HTML, PDF, LaTeX, etc.
   │   └── figures         <- Generated graphics and figures to be used in reporting
   │
   ├── environment.yml     <- yaml file for reproducing the conda analysis environment
   │
   ├── setup.py            <- makes src pip installable (pip install -e .)
   ├── src                 <- Source code for use in this project.
   │   ├── __init__.py     <- Makes src a Python module
   │   │
   │   ├── prepare_data.py <- Codes for generating the processed data from the raw data
   │   │
   │   └── utils.py        <- Utility codes, including functions for processing data
   ├── LICENSE
   └── README.md

Contents
========

.. toctree::
   :maxdepth: 2

   notebooks_link
   getting_started
   known_issues
   data_preparation
   notebooks
   api
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
