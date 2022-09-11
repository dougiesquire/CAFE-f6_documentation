.. _Producing:

Producing these docs
====================

.. toctree::
   :maxdepth: 1
   :hidden:

   data_preparation
   skill_preparation
   API_reference

These pages include information on how to generate all of the figures and analysis in these docs using Australia's National Computational Infrastructure. To reproduce the steps described below, users will need to have access to projects xv83, oi10, and ua8 which house the CAFE-f6, CMIP6 and reference datasets. Below, we describe how to set up the analysis environment, how to pre-process the data for assessment and how to perform the skill assessment. Auto-generated documentation of the Python code used to carry out these tasks is also provided.


Documentation repo structure
----------------------------

This documentation was generated using `Sphinx <https://www.sphinx-doc.org/en/master/>`_ from a `public repository hosted on github <https://github.com/dougiesquire/CAFE-f6_documentation>`_. This repository contains all of this documentation, as well as all the (Python) code used to carry out the analyses presented in this documentation (e.g. the :ref:`skill assessment <Assessment>`). The structure of the repository is as follows:

::

   ├── Makefile            <- Makefile with commands like `make data`
   ├── data
   │   ├── processed       <- The postprocessed data
   │   ├── exploratory     <- Messy, uncurated data generated while exploring things
   │   ├── skill           <- Skill assessment of processed data
   │   ├── raw             <- The original, immutable data (or symlinks to them)
   │   └── testing         <- The data used while checking CAFE-f6 forecast reproducibility
   │
   ├── config
   │   ├── prepare_data    <- Configuration files for processing the raw data
   │   └── verify          <- Configuration files for skill assessment of processed data
   │
   ├── docs                <- The Sphinx documentation
   │
   ├── notebooks           <- Jupyter notebooks containing analyses.
   │
   ├── environment.yml     <- yaml file for reproducing the conda analysis environment
   │
   ├── shell               <- Shell scripts used in this project
   │
   ├── setup.py            <- makes src pip installable (pip install -e .)
   ├── src                 <- Source code for use in this project.
   │   ├── __init__.py     <- Makes src a Python module
   │   ├── prepare_data.py <- Codes for generating the processed data from the raw data
   │   ├── verify.py       <- Codes for assessing the skill of the processed data
   │   ├── plot.py         <- Plotting codes
   │   └── utils.py        <- Utility codes, including functions for processing data
   │
   ├── CITATION.cff.       <- Information on how to cite this documentation
   ├── LICENSE
   └── README.md


Using :code:`make`
------------------

Key steps for getting set up and processing the data are handled using `GNU make <https://www.gnu.org/software/make/>`_. From the root directory of this repository:

#. :code:`make environment` creates the python environment or updates it if it exists.
#. :code:`make data` prepares the raw data (in :code:`data/raw`) for subsequent analysis. The prepared data are stored in :code:`data/processed`. See :ref:`Data preparation`.
#. :code:`make skill` calculates the skill metrics from the processed data. See :ref:`Calculating skill metrics`.
#. :code:`make docs` rebuilds this documentation.
#. :code:`make clean` cleans up unneeded files and directories.
#. :code:`make lint` runs `black <https://github.com/psf/black>`_ and `flake8 <https://github.com/PyCQA/flake8>`_ on :code:`src`.


Code documentation
------------------

All the Python code used to carry out the analyses presented in this documentation is contained in a Python package :code:`src` which is installed when the analysis environment is set up. :code:`src` contains submodules for data preparation, verification, utilities and plotting. Auto-generated API reference documentation for :code:`src` and its submodules has been auto-generated and is available at :ref:`API reference <API>`. 
