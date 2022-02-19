Assessment of CAFE-f6 hindcasts/forecasts
==============================

[![Documentation Status](https://readthedocs.org/projects/cafef6/badge/?version=latest)](https://cafef6.readthedocs.io/en/latest/?badge=latest)

Skill benchmarking of the CAFE-f6 decadal hindcast/forecast dataset

## Project Organization

    ├── LICENSE
    ├── Makefile            <- Makefile with commands like `make data`
    ├── README.md           <- The top-level README for developers using this project.
    ├── data
    │   ├── config          <- Configuration files for processing the raw data
    │   ├── processed       <- The postprocessed data
    │   ├── raw             <- The original, immutable data (or symlinks to them)
    │   └── testing         <- The data used while checking CAFE-f6 forecast reproducibility
    │
    ├── docs                <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── notebooks           <- Jupyter notebooks containing analyses. Numbers are used for ordering
    │                          where appropriate
    │
    ├── references          <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports             <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures         <- Generated graphics and figures to be used in reporting
    │
    ├── environment.yml     <- The environment file for reproducing the conda analysis environment
    │
    ├── setup.py            <- makes project pip installable (pip install -e .) so src can be imported
    └──src                 <- Source code for use in this project.
        ├── __init__.py     <- Makes src a Python module
        │
        ├── prepare_data.py <- Codes for generating the processed data from the raw data
        │
        └── utils.py        <- Utility codes, including processing methods required to generate the 
                               processed data
     

--------

