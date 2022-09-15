.. _Overview:

Overview
========

.. note::
   Hereafter, we use "forecast" to refer generally to both hindcasts and forecasts.


The CAFE-f6 model
-----------------

The CAFE-f6 decadal forecasts were generated using the Climate Analysis Forecast Ensemble (CAFE) near-term climate prediction system that was developed by the Commonwealth Science and Industrial Research Organization (CSIRO) Decadal Climate Forecast Project (DCFP). The system uses the Geophysical Fluid Dynamics Laboratory Coupled Model version 2.1 (GFDL CM2.1) :footcite:p:`delworth2006gfdl`, with an upgraded ocean model component. In the version of CAFE used by CAFE-f6, the ocean was modelled using GFDL MOM 5.1. The nominal resolution of ocean model is 1° (with increased resolution in the tropics and Southern Ocean) on 50 vertical levels. The atmospheric model (AM2) has a resolution of 2° in latitude and 2.5° in longitude on 24 hybrid vertical levels. The sea-ice model (SIS) and land model (LM2) are on the same horizontal grids as the ocean and atmospheric models, respectively. A detailed description of the CAFE modelling system can be found in :footcite:t:`o2019coupled`, :footcite:t:`o2021cafe60v1a` and :footcite:t:`o2021cafe60v1b`.


Generation of forecasts
-----------------------

Each forecast in the CAFE-f6 dataset comprises a 96-member ensemble of ten-year integrations from realistic initial states with prescribed external forcing. Forecasts have been initialised at the beginning of every May and November over the period 1981-2020. Full-field initial conditions for the 96 forecast members were taken directly from the 96-member climate reanalysis, CAFE60v1, which was also generated using the CAFE system (however, using MOM 4.1 for the ocean, rather than MOM 5.1). Details and evaluation of CAFE60v1 can be found in :footcite:t:`o2021cafe60v1a` and :footcite:t:`o2021cafe60v1b`. Note that a number of methodological changes were made during the generation of CAFE60v1 that resulted in systematic changes to the reanalysis that are also observed in the CAFE-f6 forecasts - see `Change in CAFE60v1 bias correction scheme`_.

.. _Change in CAFE60v1 bias correction scheme: notebooks/issues_bias.ipynb

Prescribed forcing fields are based on those used for the GFDL CM2.1 submissions to the Coupled Model Intercomparison Project (CMIP), phases 3 and 5 :footcite:p:`zhang2017estimating`. However, some forcing components are applied in a way that is inconsistent with CMIP Decadal Climate Prediction Project (DCPP) specifications. Specifically, a number of the forcing components switch from time-varying to fixed based on the *initialisation date* of the forecast. This means that the same calendar year can experience different forcing, depending on its lead time. See `Application of forcing`_ for more details.

.. _Application of forcing: notebooks/issues_forcing.ipynb


Accessing the data
------------------

Currently, CAFE-f6 data are only available to users of Australia's National Computational Infrastructure (NCI). Each forecast is stored as a set of zipped zarr collections within the xv83 project at :code:`/g/data/xv83/dcfp/CAFE-f6`. These collections can be efficiently read and analysed using Python and xarray. For example, to open the forecast of monthly ocean variables initialised on 2020-11-01:

.. code-block:: python

  import xarray as xr
  
  forecast_path = "/g/data/xv83/dcfp/CAFE-f6/c5-d60-pX-f6-20201101/ocean_month.zarr.zip"
  ds = xr.open_dataset(forecast_path, engine="zarr", chunks={})
  
Or to open and stack all CAFE-f6 monthly ocean forecasts for analysis (note, the following code opens, but doesn't load, almost 400 TB of data in a matter of seconds thanks to the magic of xarray + dask + zarr):

.. code-block:: python

  import xarray as xr


  def convert_time_to_lead(ds):
      """
      Return provided xarray object with time dimension converted to initial/lead time
      dimensions and time added as additional coordinate

      Parameters
      ----------
      ds : xarray Dataset
          A dataset with a time dimension
      """
      init_date = ds["time"][0].item()
      freq = xr.infer_freq(ds["time"])
      lead_time = range(len(ds["time"]))
      time_coord = (
          ds["time"]
          .rename({"time": "lead"})
          .assign_coords({"lead": lead_time})
          .expand_dims({"init": [init_date]})
      ).compute()
      dataset = ds.rename({"time": "lead"}).assign_coords(
          {"lead": lead_time, "init": [init_date]}
      )
      dataset = dataset.assign_coords({"time": time_coord})
      dataset["lead"].attrs["units"] = freq
      return dataset


  forecast_dir = "/g/data/xv83/dcfp/CAFE-f6"
  realm = "ocean_month"
  
  ds = xr.open_mfdataset(
      f"{forecast_dir}/c5-d60-pX-f6-*/{realm}.zarr.zip",
      preprocess=convert_time_to_lead,
      compat="override",
      coords="minimal",
      engine="zarr",
      parallel=True,
  )

.. note::
   The above code blocks will only work for members of the xv83 project.


Citing the data
---------------

If you use CAFE-f6 and/or any of the contents of this repository, please cite them using the metadata in `this CITATION.cff file <https://github.com/dougiesquire/CAFE-f6_documentation/blob/main/CITATION.cff>`_ or by navigating to the `base repository for this documentation <https://github.com/dougiesquire/CAFE-f6_documentation>`_ and clicking "Cite this repository" at the top right of the page.


References
----------

.. footbibliography::
