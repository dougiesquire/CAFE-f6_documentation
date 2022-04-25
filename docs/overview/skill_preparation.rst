Calculating skill metrics
=========================

Configuration files
-------------------

Steps for calculating skill metrics from a set of prepared datasets are also specified in yaml files stored in ``config/verify``. Here is an example config yaml file for preparing the initialised component of the correlation skill of CAFE-f6 relative to HadISST:

.. code-block:: yaml

   prepare:
     CAFEf6.HadISST.annual.anom_1991-2020.sst.ri:         # <- Unique identifier for the
                                                          #    metric being calculated.
                                                          #    This will be used to save
                                                          #    the metric.
       hindcast: CAFEf6.annual.anom_1991-2020.sst         # <- The name of the prepared
                                                          #    hindcast data to verify.
       observation: HadISST.annual.anom_1991-2020.sst     # <- The name of the prepared
                                                          #    observation data to verify
                                                          #    against.
       reference: CAFE_hist.annual.anom_1991-2020.sst     # <- The name of the prepared
                                                          #    historical data to use as
                                                          #    a baseline. Alternatively
                                                          #    users can specify
                                                          #    "climatology" or 
       verify:                                            #    "persistence" baselines
         metric: "acc_initialised"                        # <- The name of the metric. A
                                                          #    corresponding method must
                                                          #    exist in src.verify.
         significance: True                               # <- Whether or not to block
                                                          #    boostrap for significant
                                                          #    points using method from
                                                          #    Goddard et al. (2013)
         transform: "Fisher_z"                            # <- Transform to apply when
                                                          #    determining significant
                                                          #    points
         alpha: 0.1                                       # <- Confidence level for
                                                          #    assigning significance

Code for preparing a skill metric from a specified yaml file is in ``src/verify.py``:

.. code-block:: console

   $ python src/verify.py -h
   usage: verify.py [-h] [--config_dir CONFIG_DIR] [--save_dir SAVE_DIR] config
   
   Prepare skill metrics according to a provided config file
   
   positional arguments:
     config                Configuration file to process
   
   optional arguments:
     -h, --help            show this help message and exit
     --config_dir CONFIG_DIR
                           Location of directory containing config file(s) to use,
                           defaults to <project_dir>/config/verify
     --save_dir SAVE_DIR   Location of directory to save skill data to, defaults to
                           <project_dir>/data/skill/

The process for preparing a particular skill metric or set of metrics from a config file is very similar to the process for preparing data:

.. code-block:: console

   make skill config=<name-of-config>

This will submit a batch job to calculate all of the metrics specified in ``config/skill/<name-of-config>``. An output file (named ``skill_<name-of-config>.o????????``) for this batch job will be written to the current directory once this job is complete. Alternatively, users can process multiple datasets in multiple jobs with:

.. code-block:: console

   make skill config="<name-of-config-1> <name-of-config-2>"

or process all available datasets with:

.. code-block:: console

   make skill

Adding a new skill metric
-------------------------
There are a few steps to adding a new skill metric.

#. Prepare a config file for the new metric or add the metric to an existing config file. Note that the file can be named anything you like. However, a function with the same name as any skill metrics specified in the config file must be implemented in ``src.verify``. These functions should operate on timeseries and should be verbosely named.
#. If you made a new config file, add it to the list of default configs to process (variable ``skill_config``) in ``Makefile``

