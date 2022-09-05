Issues
======

In these pages, we describe and demonstrate a number of issues with the CAFE-f6 forecasts that must be contended with when using the data. To summarise:

- The external forcing is applied in a way that is inconsistent with CMIP Decadal Climate Prediction Project (DCPP) recommendations. Users are encouraged to familiarise themselves with the issue and consider how it may impact their application of the CAFE-f6 data.

- Changes to the bias correction scheme in CAFE60v1 in 1992 (which provides the initial conditions for CAFE-f6) have a noticable and lasting impact on the CAFE-f6 forecasts. Users are encouraged assess the impact of the change in their application of the CAFE-f6 data.

- Changes to the CAFE-f6 ocean model executable were made while the CAFE-f6 dataset was being generated that impact the reproducibility of some of the May-initialised forecasts. Users are encouraged to review which forecasts are reproducible using the current executable. A simple way to avoid this issue altogether is to use only the November-initialised forecasts.

- Changes to the atmospheric model timestep for some ensemble members of some forecasts produced unexpected drift in those forecasts. Users are encouraged to use only forecasts that have an atmospheric timestep of 1800 s. A mask for these forecasts can be found at `/g/data/xv83/dcfp/CAFE-f6/CAFE-f6_dt_atmos.nc`

Full details of each issue are given in the following pages:

.. toctree::
   :maxdepth: 2

   notebooks/issues_forcing.ipynb
   notebooks/issues_bias.ipynb
   notebooks/issues_executable.ipynb
   notebooks/issues_timestep.ipynb