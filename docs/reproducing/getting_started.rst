Getting started
===============

Project structure
-----------------

The structure of this project is as follows:

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
   ├── LICENSE
   └── README.md


Using `make <https://www.gnu.org/software/make/>`_
--------------------------------------------------

Key steps for getting set up are handled using `GNU make <https://www.gnu.org/software/make/>`_:

#. ``make environment`` creates the python environment or updates it if it exists
#. ``make data`` prepares the raw data (in ``data/raw``) for subsequent analysis. The prepared data are stored in ``data/processed``. See :ref:`Data preparation`
#. ``make skill`` calculates the skill metrics from the processed data. See :ref:`Calculating skill metrics`
#. ``make docs`` rebuilds this documentation
#. ``make clean`` cleans up unneeded files and directories
#. ``make lint`` runs ``black`` and ``flake8`` on ``src``
