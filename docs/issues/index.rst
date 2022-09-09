.. _Issues:

Known issues
============

.. toctree::
   :maxdepth: 1
   :hidden:

   ../notebooks/issues_forcing.ipynb
   ../notebooks/issues_bias.ipynb
   ../notebooks/issues_executable.ipynb
   ../notebooks/issues_timestep.ipynb
   ../notebooks/issues_units.ipynb

In these pages, we describe and demonstrate a number of issues with the CAFE-f6 forecasts that must be considered and contended with when using the data. The issues are summarised below, along with recommendations for how to deal with them. Click on the links for more details on each issue:

- `Application of forcing`_ - The external forcing is applied in a way that is inconsistent with CMIP Decadal Climate Prediction Project (DCPP) recommendations. Users are encouraged to familiarise themselves with the issue and consider how it may impact their application of the CAFE-f6 data.

- `Change to CAFE60v1 bias correction scheme`_ - Changes to the bias correction scheme in CAFE60v1 (which provides the initial conditions for CAFE-f6) in 1992 have a noticable and lasting impact on the CAFE-f6 forecasts. Users are encouraged assess the impact of the change in their application of the CAFE-f6 data.

- `Changes to MOM executable`_ - Changes to the CAFE-f6 ocean model executable were made while the CAFE-f6 dataset was being generated that impact the reproducibility of some of the May-initialised forecasts. Users are encouraged to review which forecasts are reproducible using the current executable. A simple way to avoid this issue altogether is to use only the November-initialised forecasts.

- `Changes to model timesteps`_ - Changes to the atmospheric model timestep for some ensemble members of some forecasts produced unexpected drift in those forecasts. Users are encouraged to use only forecasts that have an atmospheric timestep of 1800 s. A mask for these forecasts can be found at `/g/data/xv83/dcfp/CAFE-f6/CAFE-f6_dt_atmos.nc`.

- `Incorrect units`_ - During the conversion from raw model output to zarr format, units were incorrectly assigned to the `"precip"` and `"evap"` variables. Users should correct these data prior to analysis by normalising them by the number of days in each month.

.. _Application of forcing: ../notebooks/issues_forcing.ipynb
.. _Change to CAFE60v1 bias correction scheme: ../notebooks/issues_bias.ipynb
.. _Changes to MOM executable: ../notebooks/issues_executable.ipynb
.. _Changes to model timesteps: ../notebooks/issues_timestep.ipynb
.. _Incorrect units: ../notebooks/issues_units.ipynb