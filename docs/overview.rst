The CAFE-f6 hindcasts/forecasts
===============================

The model
---------

The CAFE-f6 decadal forecasts (and hindcasts) were generated using the Climate Analysis Forecast Ensemble (CAFE) near-term climate prediction system, that was developed by the Commonwealth Science and Industrial Research Organization (CSIRO) Decadal Climate Forecast Project (DCFP). The system uses the Geophysical Fluid Dynamics Laboratory Coupled Model (GFDL) version 2.1 ([`Delworth et al 2006`]_), with an upgraded oceanic component. In the version of CAFE used by CAFE-f6, the ocean was modelled using GFDL MOM 5.1. The nominal resolution of ocean model is 1° (with higher latitudinal and longitudinal resolution in the tropics and Southern Ocean, respectively) on 50 vertical levels. The atmospheric model (AM2) has a resolution of 2° in latitude and 2.5° in longitude on 24 hybrid vertical levels. The sea-ice model (SIS) and land model (LM2) are on the same horizontal grids as the ocean and atmospheric models, respectively. A detailed description of the CAFE modelling system can be found in [O'Kane_et_al_2018]_, [O'Kane_et_al_2021a]_ and [O'Kane_et_al_2021b]_.

Hereafter, we use "forecast" to denote both hindcasts and forecasts.

Generation of forecasts
-----------------------

Each forecast in the CAFE-f6 dataset comprises a 96-member ensemble of ten-year integrations from realistic initial states with prescribed external forcing. Forecasts have been initialised at the beginning of every May and November over the period 1981-2020. Full-field initial conditions for the 96 forecast members were taken directly from the 96-member climate reanalysis, CAFE60v1, which was also generated using the CAFE system (however, using MOM 4.1 for the ocean, rather than MOM 5.1). Details and evaluation of CAFE60v1 can be found in [O'Kane_et_al_2021a]_ and [O'Kane_et_al_2021b]_. Note that a number of methodological changes were made during the generation of CAFE60v1 that resulted in systematic changes to the reanalysis that are also observed in the CAFE-f6 forecasts - see `Issues with CAFE-f6`_.

.. _Issues with CAFE-f6: assessment/notebooks/CAFE-f6_issues.ipynb

Prescribed forcing fields are based on those used for the GFDL CM2.1 submissions to the Coupled Model Intercomparison Project (CMIP), phases 3 and 5 ([Zhang_et_al_2017]).

Forcing
-------

Accessing the data
------------------

Currently, CAFE-f6 data is only available to users of Australia's National Computational Infrastructure (NCI). The data is stored as

References
----------

.. [`Delworth et al_2006`] Delworth, T. L. et al. GFDL’s CM2 global coupled climate models. Part I: formulation and simulation characteristics. J. Clim. 19, 643–674 (2006).

.. [O'Kane_et_al_2018] O’Kane, T. J. et al. Coupled data assimilation and ensemble initialization with application to multiyear ENSO prediction. J. Clim. 32, 997–1024 (2018).

.. [O'Kane_et_al_2021a] O’Kane, T. J. et al. CAFE60v1: a 60-year large ensemble climate reanalysis. Part I: system design, model configuration and data assimilation. J. Clim. 1, 1–48 (2021).

.. [O'Kane_et_al_2021b] O’Kane, T. J. et al. CAFE60v1: a 60-year large ensemble climate reanalysis. Part II: evaluation. J. Clim. 1, 1–62 (2021).

.. [Zhang_et_al_2017] Zhang, L. et al. Estimating decadal predictability for the Southern Ocean using the GFDL CM2.1 model. J. Clim, 30, 5187–5203 (2017).
