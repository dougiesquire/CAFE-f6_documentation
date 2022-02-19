Getting started
===============

This project is organised as follows:

::
   ├── Makefile            <- Makefile with commands like `make data`
   ├── data
   │   ├── config          <- Configuration files for processing the raw data
   │   ├── processed       <- The postprocessed data
   │   ├── raw             <- The original, immutable data (or symlinks to them)
   │   └── testing         <- The data used while checking CAFE-f6 forecast reproducibility
   │
   ├── docs                <- Sphinx documentation
   │
   ├── notebooks           <- Jupyter notebooks containing analyses. Numbers are used for ordering where appropriate
   │
   ├── references          <- Data dictionaries, manuals, and all other explanatory materials.
   │
   ├── reports             <- Generated analysis as HTML, PDF, LaTeX, etc.
   │   └── figures         <- Generated graphics and figures to be used in reporting
   │
   ├── environment.yml     <- The environment file for reproducing the conda analysis environment
   │
   ├── setup.py            <- makes project pip installable (pip install -e .) so src can be imported
   ├──src                 <- Source code for use in this project.
   │   ├── __init__.py     <- Makes src a Python module
   │   │
   │   ├── prepare_data.py <- Codes for generating the processed data from the raw data
   │   │
   │   └── utils.py        <- Utility codes, including processing methods required to generate the processed data
   ├── README.md
   └── LICENSE
Key steps for getting set up are handled using `make <https://www.gnu.org/software/make/>`_:

#. ``make environment`` creates the python environment or updates it if it exists
#. ``make data`` prepares the raw data (in ``data/raw``) for subsequent analysis. The processed data are stored in ``data/processed``. See :ref:`Data Preparation`
#. ``make docs`` rebuilds this documentation
#. ``make clean`` cleans up unneeded files and directories
#. ``make lint`` runs ``black`` and ``flake8`` on ``src``
