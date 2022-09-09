.. _Assessment:

Skill assessment
================

.. toctree::
    :maxdepth: 1
    :hidden:

    ../notebooks/assessment_generic.ipynb
    ../notebooks/assessment_Aus.ipynb

These pages present some preliminary assessment of the skill of the CAFE-f6 hindcasts. The assessment approach largely follows the assessment of the CanESM5 decadal hindcasts in :footcite:t:`sospedra2021decadal`. Skill scores are computed for the CAFE-f6 hindcasts, and for the (40 ensemble member) CanESM5 :footcite:p:`sospedra2021decadal` and (10 ensemble member) EC-Earth3 :footcite:p:`bilbao2021assessment` CMIP6 Decadal Climate Prediction Project (DCPP) hindcast submissions for comparison. Note that the CanESM5 hindcasts are initialised at the end of December every year, while the CAFE-f6 and EC-Earth3 hindcasts are initialised at the beginning of November every year (only Novemeber-initialised CAFE-f6 forecasts are considered here due to the `reproducibility issue`_ with many of the May-initialised forecasts).

.. _reproducibility issue: ../notebooks/issues_executable.ipynb


Forced historical simulations
-----------------------------

For each hindcast dataset, a forced historical simulation is used to help quantify the skill added by initialisation of the hindcasts relative to the uninitialised simulations. For the CMIP6 DCPP hindcasts, historical simulations are taken from the corresponding CMIP6 "historical" experiment, ensuring that the same number of ensemble members are used for each. For the CAFE-f6 hindcasts, a dedicated 96-member forced historical simulation and corresponding 20-member control simulation has been generated (see `here <https://github.com/dougiesquire/cm-historical>`_ for the run scripts). These simulations were initialised from CAFE60v1 in 1960-11-01 and run for 80 years, with the assumption that ensemble members will be independent by the beginning of the CAFE-f6 hindcast period (1981). The CAFE historical simulation experiences the same external forcing as the CAFE-f6 hindcasts *at initialisation* (see `Application of forcing`_) and the control simulation experiences fixed 1960 forcing. The ensemble-mean control run climatology is substracted from the historical run to account for model drift over the relative short simulation period.

.. _Application of forcing: ../notebooks/issues_forcing.ipynb


Skill assessment methods
------------------------

Unless otherwise specified, skill assessment is performed on hindcast anomalies that are computed relative to each model's own ensemble-mean climatology as a function of lead time. 30-year climatological and verification periods are used for both the CAFE-f6 and DCPP data. However, because the historical CMIP6 data end in 2014, these periods differ slightly for the different model hindcasts: 1991-2020 for CAFE-f6; 1985-2014 for CanESM5 and EC-Earth3. Anomalies of the reference data (the "truth" to verify the hindcasts against, e.g. observations) are computed relative to their climatological mean over the same period as the hindcast data. All model and reference data are bi-linearly interpolated to the CAFE-f6 atmospheric grid prior to the calculation of skill scores.

The following reference datasets are used to assess the hindcast skill:

.. list-table:: Reference datasets
   :widths: 65 35
   :header-rows: 1

   * - Variable(s)
     - Reference dataset
   * - Global SST, NINO 3.4, DMI, AMV, IPO
     - HadISST :footcite:p:`rayner2003global`
   * - Global upper ocean heat content
     - EN.4.2.2 c14 :footcite:p:`good2013en4`
   * - Global 2m temperature, global SLP, SAM, NAO, Australian 10m wind, Australian FFDI
     - JRA55 :footcite:p:`kobayashi2015jra,harada2016jra`
   * - Global precipitation
     - GPCP v02r03 :footcite:p:`adler2018global`
   * - Australian 2m temperature, Australian extreme 2m temperature, Australian precipitation, Australian extreme precipitation, Australian drought, Australian EHF severity
     - AGCDv2 :footcite:p:`evans2020enhanced`
     
A number of deterministic and probabilistic skill metrics were calculated, with only a few deterministic metrics, computed on the ensemble-mean, shown in this documentation. Following :footcite:t:`sospedra2021decadal`, the reference anomalies are denoted as :math:`X`, the ensemble-mean hindcast anomalies as :math:`Y`, the ensemble-mean simulation anomalies as :math:`U`, :math:`C_{AB}` denotes the covariance of :math:`A` and :math:`B`, and :math:`\sigma_{A}` denotes the standard deviation of :math:`A`. The following table defines the skill metrics presented in this documentation.

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
     - :math:`\mathrm{MSSS}(Y, R, X) = 1 - \frac{\mathrm{MSE}(Y, X)}{\mathrm{MSE}(R,X)}` where :math:`\mathrm{MSE}` is the mean square error and :math:`R` corresponds to baseline predictions of either observed climatology, persistence or uninitialised simulations.
     - Is the forecast error smaller than a baseline prediction?
     - :footcite:t:`goddard2013verification`

Statistical significance of the skill scores is evaluated using a non-parametric cicular moving-block bootstrap approach :footcite:p:`sospedra2021decadal, goddard2013verification`. Following :footcite:t:`sospedra2021decadal`, 1000 repetitions are performed using 5-year blocks. Skill scores that are found to be significant at the 95\% confidence level are indicated in the following pages (using hatching on map plots and dots on line plots). 

.. note::
   We define the "0th" lead period of a forecast as the period that includes the initialisation. For example, for annual forecasts initialised on 2022-11-01, "lead year 0" refers to the period 2022-11-01 - 2023-10-31, "lead year 1" refers to 2023-11-01 - 2024-20-31. This is different than some existing studies (e.g. :footcite:t:`sospedra2021decadal`) whose "lead year 1" is equivalent to our "lead year 0".


Hindcast skill results
----------------------

The skill assessment is split into two sections:

- `Generic hindcast skill`_ - Assessment of the skill of commonly assessed quantities, including gridded global variables and indices for key climate drivers.

- `Australian hindcast skill`_ - Assessment of the skill of Australian regionally-averaged quantities. This analysis has some focus on climate extremes and was motivated by deliverables for the Australian Climate Service.

.. _Generic hindcast skill: ../notebooks/assessment_generic.ipynb
.. _Australian hindcast skill: ../notebooks/assessment_Aus.ipynb

Documentation of the code and workflows used to carry out these skill assessment can be found in :ref:`Producing these docs <Producing>`.

References
----------

.. footbibliography::
