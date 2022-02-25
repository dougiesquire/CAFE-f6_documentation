Data Preparation
================

Steps for preparing the various datasets used in this project are specified in yaml files stored in ``data/config``. Here is an example config yaml file for preparing annual precipitation for the CanESM5 CMIP6 DCPP submissions:

.. code-block:: yaml

   # name: required
   #    The name of the dataset. Must match a method in src.prepare_data._open
   # prepare: required
   #    Output variables to prepare and save. Each variable can include the following
   #    identifier: required
   #        Unique identifier for the output variable being processed. This will be
   #        used to save this diagnostics as: {name}.{identifier}.zarr.
   #    uses: required
   #        List of input variables required to compute the output variable. For some
   #        datasets, this should be further broken into subkeys indicating the realm
   #        for each list of variables (e.g. ocean_month). Alternatively, users can
   #        provide the identifier of a previously prepared dataset by entering
   #        `prepared: <identifier>`.
   #    preprocess: optional
   #        Functions and kwargs from src.utils to be applied sequentially prior to
   #        concatenation (for datasets comprised of multiple concatenated) and/or
   #        prior to merging of variables from multiple realms into a single dataset.
   #        These are applied before the variables are renamed and converted.
   #    apply: optional
   #        Functions and kwargs from src.utils to be applied sequentially to opened
   #        dataset. These are applied after the variables are renamed and converted.

   name: "CanESM5"

   prepare:
     precip:
       identifier: "precip.annual.full"
       uses:
         Amon:
           - "pr"
       apply:
         rename:
           names:
             pr: "precip"
         convert:
           conversion:
             precip:
               multiply_by: 86400
         coarsen_monthly_to_annual:
           dim: "lead"
         interpolate_to_grid_from_file:
           file: "data/raw/gridinfo/CAFE_atmos_grid.nc"
         rechunk:
           chunks: {"init": -1, "lead": 1, "member": -1, "lat": -1, "lon": -1}
   
Code for preparing data from a specified yaml file is in ``src/prepare_data.py``:

.. code-block:: console

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

This will submit a batch job to prepare all of the diagnositics specified in ``data/config/<name-of-config>``. An output file (named ``make_<name-of-config>.o????????``) for this batch job will be written to the current directory once this job is complete. Alternatively, users can process multiple datasets in multiple jobs with:

.. code-block:: console

   make data config="<name-of-config-1> <name-of-config-2>"

or process all available datasets with:

.. code-block:: console

   make data

Adding a new dataset for preparation
------------------------------------
There are a few steps to adding a new dataset.

#. Add a step to the 'data' trigger within ``Makefile`` symlinking the location of the data in ``data/raw``. (This is really just to keep things tidy/easily-traceable.)
#. Add a new, appropriately-named, method to ``src/prepare_data._open``. Choose a name that uniquely identifies the dataset being added, e.g. "JRA55".
#. Prepare a config file for the new dataset. This file can be named anything, however, the 'name' key must match the name of the new method added in 2. Functions for executing new steps should be added to ``src/utils.py``.
#. Add the new config file to the list of default configs to process (variable ``configs``) in ``Makefile``
