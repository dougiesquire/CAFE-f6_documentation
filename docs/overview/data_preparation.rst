Data preparation
================

Steps for preparing the various datasets used in this project are specified in yaml files stored in ``config/prepare_data``. Here is an example config yaml file for preparing annual precipitation and its anomalies for the CAFE-f6 hindcasts/forecasts:

.. code-block:: yaml

   name: "CAFEf6"                       # <- The name of the dataset. This must match
                                        #    the name of a corresponding method in
                                        #    src.prepare_data._open

   prepare:
     annual.full.precip:                # <- Unique identifier for the output variable
                                        #    being processed. This will be used to save
                                        #    the variable as {name}.{identifier}.zarr
       uses:                            # <- List of input variables required to compute
         atmos_isobaric_month:          #    the output variable. For some datasets this
           - "precip"                   #    should be futher broken down into subkeys
                                        #    indicating the realm for each list of
                                        #    variables (e.g. atmos_isobaric_month).
                                        #    Users can also provide the identifier of a
                                        #    previously prepared variable using:
                                        #      uses:
                                        #        prepared:
                                        #          - <identifier> 
       preprocess:                      # <- Functions and kwargs from src.utils to be
         normalise_by_days_in_month:    #    applied sequentially prior to concatenation
         convert_time_to_lead:          #    (for datasets comprised of multiple
         truncate_latitudes:            #    concatenated files) and/or prior to merging
         coarsen_monthly_to_annual:     #    input variables from multiple realms where 
           dim: "lead"                  #    more than one are specified
       apply:                           # <- Functions and kwargs from src.utils to be
         rename:                        #    applied sequentially to the opened (and
           names:                       #    concatenated/merge, where appropriate)
             ensemble: "member"         #    dataset
         convert:
           conversion:
             precip:
               multiply_by: 86400
         round_to_start_of_month:
           dim: ["init", "time"]
         rechunk:
           chunks: {"init": -1, "lead": 2, "member": -1, "lat": 45, "lon": 72}

     annual.anom_1991-2020.precip:
       uses:
         prepared:
           - "annual.full.precip"
       apply:
         anomalise:
           clim_period: ["1991-01-01", "2020-12-31"]
         rechunk:
           chunks: {"init": -1, "lead": 2, "member": -1, "lat": 45, "lon": 72}

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
                           defaults to <project_dir>/config/prepare_data/
     --save_dir SAVE_DIR   Location of directory to save processed data to, defaults to
                        <project_dir>/data/processed/

To prepare a particular dataset, run:

.. code-block:: console

   make data config=<name-of-config>

This will submit a batch job to prepare all of the diagnositics specified in ``config/prepare_data/<name-of-config>``. An output file (named ``data_<name-of-config>.o????????``) for this batch job will be written to the current directory once this job is complete. Alternatively, users can process multiple datasets in multiple jobs with:

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
#. Add the new config file to the list of default configs to process (variable ``data_config``) in ``Makefile``
