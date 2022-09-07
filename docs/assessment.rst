Assessment
==========

These pages present some preliminary assessment of the skill of the CAFE-f6 forecasts. The assessment approach largely follows the assessment of the CanESM5 decadal hindcasts in :footcite:t:`sospedra2021decadal`. Skill scores are computed for the CAFE-f6 hindcasts, and for the (40 ensemble member) CanESM5 :footcite:p:`sospedra2021decadal` and (10 ensemble member) EC-Earth3 :footcite:p:`bilbao2021assessment` CMIP6 Decadal Climate Prediction Project (DCPP) hindcast submissions for comparison. 


Forced historical simulations
-----------------------------

For each hindcast dataset, a forced historical simulation is used to help quantify the skill added by initialisation of the hindcasts relative to the uninitialised simulations. For the CMIP6 DCPP hindcasts, historical simulations are taken from the corresponding CMIP6 "historical" experiment, ensuring that the same number of ensemble members are used for each [#]_. For the CAFE-f6 hindcasts, a dedicated 96-member forced historical simulation and corresponding 20-member control simulation has been generated (see `here <https://github.com/dougiesquire/cm-historical>`_ for the run scripts). These simulations were initialised from CAFE60v1 in 1960-11-01 and run for 80 years, with the assumption that ensemble members will be independent by the beginning of the CAFE-f6 hindcast period (1981). The CAFE historical simulation experiences the same external forcing as the CAFE-f6 hindcasts *at initialisation* (see `Application of forcing`_) and the control simulation experiences fixed 1960 forcing. The ensemble-mean control run climatology is substracted from the historical run to account for model drift over the relative short simulation period.

.. _Application of forcing: notebooks/issues_forcing.ipynb


Evaluation methods
------------------

Unless otherwise specified, skill assessment is performed on hindcast anomalies that are computed relative to each model's own ensemble-mean climatology as a function of lead time. 30-year climatological and verification periods are used for both the CAFE-f6 and DCPP data. However, because the historical CMIP6 data end in 2014, these periods differ slightly for the different model hindcasts: 1991-2020 for CAFE-f6; 1985-2014 for CanESM5 and EC-Earth3 [#]_. Anomalies of the reference data (the "truth" to verify the hindcasts against, e.g. observations) are computed relative to their climatological mean over the same period as the hindcast data. All model and reference data are bi-linearly interpolated to the CAFE-f6 atmospheric grid prior to the calculation of skill scores.

A number of deterministic and probabilistic skill metrics were calculated, with only a few deterministic metrics, computed on the ensemble-mean, shown here. Following :footcite:t:`sospedra2021decadal`, the reference anomalies are denoted as :math:`X`, the ensemble-mean hindcast anomalies as :math:`Y`, the ensemble-mean simulation anomalies as :math:`U`, :math:`C_{AB}` denotes the covariance of :math:`A` and :math:`B`, and :math:`\sigma_{A}` denotes the standard deviation of :math:`A`. The following table defines most of the skill metrics presented in this documentation.

.. list-table:: Skill metrics
   :widths: 30 35 35 25
   :header-rows: 1

   * - Metric
     - Definition
     - Interpretation
     - Reference
   * - Anomaly cross correlation (ACC)
     - :math:`r_{XY} = \frac{C_{XY}}{\sigma_{X}\sigma_{Y}}`
     - How in phase are the hindcasts and observations?
     - :footcite:t:`wilks2011statistical`
   * - Initialised component of the ACC
     - :math:`r_{i} = r_{XY} - \theta * r_{XU} * r_{YU}` where :math:`\theta =` 0 if :math:`r_{YU} < 0` else 1
     - How much of the correlation skill came from initialisation?
     - :footcite:t:`sospedra2020assessing`
   * - Mean Squared Skill Score
     - :math:`MSSS(Y, R, X) = 1 - \frac{MSE(Y, X)}{MSE(R,X)}` where :math:`MSE` is the mean square error and :math:`R` corresponds to reference predictions of either observed climatology, persistence or uninitialised simulations.
     - Is the forecast error smaller than a baseline prediction?
     - :footcite:t:`goddard2013verification`

Statistical significance of the skill scores is evaluated using a non-parametric cicular moving-block bootstrap approach :footcite:p:`sospedra2021decadal, goddard2013verification`. Following :footcite:t:`sospedra2021decadal`, 1000 repetitions are performed using 5-year blocks. Skill scores that are found to be significant at the 95\% confidence level are indicated in the following pages (usually via hatching). 

.. note::
   We defined the "0th" lead period of a forecast as the period that includes the initialisation. For example, for annual forecasts initialised on 2022-11-01, "lead year 0" refers to the period 2022-11-01 - 2023-10-31, "lead year 1" refers to 2023-11-01 - 2024-20-31. This is different than some existing studies (e.g. :footcite:t:`sospedra2021decadal`) whose "lead year 1" is equivalent to our "lead year 0".

The following pages present skill maps for a number of global and regional variables and processes:

.. toctree::
    :maxdepth: 2

    notebooks/assessment_CanESM5.ipynb
    notebooks/assessment_global.ipynb
    notebooks/assessment_Aus.ipynb


.. rubric:: Footnotes

.. [#] Ensemble members 1, 2, 4, 6, 9, 10, 12, 14, 16, 17 are used from the EC-Earth3 historical run :footcite:p:`bilbao2021assessment`.
.. [#] Note that the CanESM5 hindcasts are initialised at the end of December every year, while the CAFE-f6 and EC-Earth3 hindcasts are initialised at the beginning of November every year.