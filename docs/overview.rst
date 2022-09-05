Overview of CAFE-f6
===================

.. note::
   Hereafter, we use "forecast" to refer generally to both hindcasts and forecasts.

The model
---------

The CAFE-f6 decadal forecasts were generated using the Climate Analysis Forecast Ensemble (CAFE) near-term climate prediction system, that was developed by the Commonwealth Science and Industrial Research Organization (CSIRO) Decadal Climate Forecast Project (DCFP). The system uses the Geophysical Fluid Dynamics Laboratory Coupled Model (GFDL) version 2.1 :footcite:p:`delworth2006gfdl`, with an upgraded oceanic component. In the version of CAFE used by CAFE-f6, the ocean was modelled using GFDL MOM 5.1. The nominal resolution of ocean model is 1° (with higher latitudinal and longitudinal resolution in the tropics and Southern Ocean, respectively) on 50 vertical levels. The atmospheric model (AM2) has a resolution of 2° in latitude and 2.5° in longitude on 24 hybrid vertical levels. The sea-ice model (SIS) and land model (LM2) are on the same horizontal grids as the ocean and atmospheric models, respectively. A detailed description of the CAFE modelling system can be found in :footcite:t:`o2019coupled`, :footcite:t:`o2021cafe60v1a` and :footcite:t:`o2021cafe60v1b`.

Generation of forecasts
-----------------------

Each forecast in the CAFE-f6 dataset comprises a 96-member ensemble of ten-year integrations from realistic initial states with prescribed external forcing. Forecasts have been initialised at the beginning of every May and November over the period 1981-2020. Full-field initial conditions for the 96 forecast members were taken directly from the 96-member climate reanalysis, CAFE60v1, which was also generated using the CAFE system (however, using MOM 4.1 for the ocean, rather than MOM 5.1). Details and evaluation of CAFE60v1 can be found in :footcite:t:`o2021cafe60v1a` and :footcite:t:`o2021cafe60v1b`. Note that a number of methodological changes were made during the generation of CAFE60v1 that resulted in systematic changes to the reanalysis that are also observed in the CAFE-f6 forecasts - see `Issues with CAFE-f6`_.

.. _Issues with CAFE-f6: assessment/notebooks/CAFE-f6_issues.ipynb

Prescribed forcing fields are based on those used for the GFDL CM2.1 submissions to the Coupled Model Intercomparison Project (CMIP), phases 3 and 5 :footcite:p:`zhang2017estimating`.

Forcing
-------

Accessing the data
------------------

Currently, CAFE-f6 data is only available to users of Australia's National Computational Infrastructure (NCI). The data is stored as

References
----------

.. footbibliography::
