Data Preparation
================

Steps for preparing the various datasets used in this project are specified in yaml files stored in `data/config`. Code for preparing data from a specified yaml file is in `src/prepare_data.py`:

::
   $ python src/prepare_data.py -h
   usage: prepare_data.py [-h] [--config_dir CONFIG_DIR] [--save_dir SAVE_DIR] config

   Process a raw dataset according to a provided config file

   positional arguments:
     config                Configuration file to process

   optional arguments:
     -h, --help            show this help message and exit
     --config_dir CONFIG_DIR
                           Location of directory containing config file(s) to use,
                           defaults to <project_dir>/data/config/
     --save_dir SAVE_DIR   Location of directory to save processed data to, defaults to
                        <project_dir>/data/processed/

To prepare a particular dataset, run:

.. code-block:: console

   make data config=<name-of-config>

This will submit a batch job to prepare all of the diagnositics specified in `data/config/<name-of-config>`. An output file (`make_<name-of-config>.o????????`) for this batch job will be written to the current directory once this job is complete. Alternatively, users can process multiple datasets in multiple jobs with:

.. code-block:: console

   make data config="<name-of-config-1> <name-of-config-2>"

or process all available datasets with:

.. code-block:: console

   make data

Adding a new dataset for preparation
------------------------------------
There are a few steps to adding a new dataset.
1. Symlink the location of the data in `data/raw`. (This is really just to keep things tidy/easily-traceable.)
2. Add a new, appropriately-named, method to `src/prepare_data._open`. Choose a name that uniquely identifies the dataset being added, e.g. "JRA55".
3. Prepare a config file for the new dataset. This file can be named anything, however, the 'name' key must match the name of the new method added in 2. Functions for executing new steps should be added to `src/utils.py`.
4. Add the new config file to the list of default configs to process in `Makefile`
