import sys
import tempfile
from pathlib import Path

import warnings

import math

from collections import OrderedDict

from itertools import chain, islice, cycle

import logging
import argparse

import dask
from dask.distributed import Client

import numpy as np

import xarray as xr
import xskillscore as xs

from src import utils


N_BOOTSTRAP_ITERATIONS = 1000

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data/processed"

dask.config.set(**{"array.slicing.split_large_chunks": False})

# Metrics
# ===============================================


def rXY(hcst, obsv, correlation="pearson_r"):
    """
    Return the anomaly cross correlation between two timeseries

    Parameters
    ----------
    hcst : xarray Dataset
        The forecast timeseries
    obsv : xarray Dataset
        The observed timeseries
    """

    if correlation == "pearson_r":
        return xs.pearson_r(hcst.mean("member"), obsv, dim="time", skipna=True)
    elif correlation == "spearman_r":
        return xs.spearman_r(hcst.mean("member"), obsv, dim="time", skipna=True)
    else:
        raise ValueError("Unrecognised value for input 'correlation'")


def ri(hcst, obsv, hist):
    """
    Return the initialised component of Pearson anomaly cross correlation
    between a forecast and observations

    Parameters
    ----------
    hcst : xarray Dataset
        The forecast timeseries
    obsv : xarray Dataset
        The observed timeseries
    hist : xarray Dataset
        The historical simulation timeseries
    """
    rXY = xs.pearson_r(hcst.mean("member"), obsv, dim="time", skipna=True)
    rXU = xs.pearson_r(obsv, hist.mean("member"), dim="time", skipna=True)
    rYU = xs.pearson_r(
        hcst.mean("member"), hist.mean("member"), dim="time", skipna=True
    )
    θ = xr.where(rYU < 0, 0, 1, keep_attrs=False)
    ru = θ * rXU * rYU
    return rXY - ru


def msss(hcst, obsv, ref, ensemble_mean=True):
    """
    Return the mean squared skill score between a forecast and observations
    relative to a reference dataset

    Parameters
    ----------
    hcst : xarray Dataset
        The forecast timeseries
    obsv : xarray Dataset
        The observed timeseries
    ref : xarray Dataset
        The reference timeseries
    ensemble_mean: boolean, optional
        If true, take the mean across the "member" dimension of hcst and ref
        (if it exists) prior to calculating the MSSS
    """
    if ensemble_mean:
        hcst = hcst.mean("member")
        ref = ref.mean("member") if "member" in ref.dims else ref

    num = xs.mse(hcst, obsv, dim="time", skipna=True)
    den = xs.mse(ref, obsv, dim="time", skipna=True)
    return 1 - num / den


def crpss(hcst, obsv, ref):
    """
    Return the Continuous rank probability skill score between a forecast and
    observations relative to a reference dataset

    Parameters
    ----------
    hcst : xarray Dataset
        The forecast timeseries
    obsv : xarray Dataset
        The observed timeseries
    ref : xarray Dataset
        The reference timeseries
    """
    num = xs.crps_ensemble(obsv, hcst, dim="time")
    den = xs.crps_ensemble(obsv, ref, dim="time")
    return 1 - num / den


# Transforms
# ===============================================


def Fisher_z(ds):
    """
    Return the Fisher-z transformation of ds

    Parameters
    ----------
    ds : xarray Dataset
        The data to apply the Fisher-z transformation to
    """
    return np.arctanh(ds)


# Bootstrapping
# ===============================================


def _get_blocked_random_indices(shape, block_axis, block_size):
    """
    Return indices to randomly sample an axis of an array in consecutive
    (cyclic) blocks
    """

    def _random_blocks(length, block):
        """
        Indices to randomly sample blocks in a cyclic manner along an axis of a
        specified length
        """
        if block == length:
            return list(range(length))
        else:
            repeats = math.ceil(length / block)
            return list(
                chain.from_iterable(
                    islice(cycle(range(length)), s, s + block)
                    for s in np.random.randint(0, length, repeats)
                )
            )[:length]

    if block_size == 1:
        return np.random.randint(
            0,
            shape[block_axis],
            shape,
        )
    else:
        non_block_shapes = [s for i, s in enumerate(shape) if i != block_axis]
        return np.moveaxis(
            np.stack(
                [
                    _random_blocks(shape[block_axis], block_size)
                    for _ in range(np.prod(non_block_shapes))
                ],
                axis=-1,
            ).reshape([shape[block_axis]] + non_block_shapes),
            0,
            block_axis,
        )


def _n_nested_blocked_random_indices(sizes, n_iterations):
    """
    Returns indices to randomly resample blocks of an array (with replacement) in
    a nested manner many times. Here, "nested" resampling means to randomly resample
    the first dimension, then for each randomly sampled element along that dimension,
    randomly resample the second dimension, then for each randomly sampled element
    along that dimension, randomly resample the third dimension etc.

    Parameters
    ----------
    sizes : OrderedDict
        Dictionary with {names: (sizes, blocks)} of the dimensions to resample
    n_iterations : int
        The number of times to repeat the random resampling
    """

    shape = [s[0] for s in sizes.values()]
    indices = OrderedDict()
    for ax, (key, (_, block)) in enumerate(sizes.items()):
        indices[key] = _get_blocked_random_indices(
            shape[: ax + 1] + [n_iterations], ax, block
        )
    return indices


def _expand_n_nested_random_indices(indices):
    """
    Expand the dimensions of the nested input arrays so that they can be broadcast
    and return a tuple that can be directly indexed

    Parameters
    ----------
    indices : list of numpy arrays
        List of numpy arrays of sequentially increasing dimension as output by the
        function `_n_nested_blocked_random_indices`. The last axis on all inputs is
        assumed to correspond to the iteration axis
    """
    broadcast_ndim = indices[-1].ndim
    broadcast_indices = []
    for i, ind in enumerate(indices):
        expand_axes = list(range(i + 1, broadcast_ndim - 1))
        broadcast_indices.append(np.expand_dims(ind, axis=expand_axes))
    return (..., *tuple(broadcast_indices))


def _iterative_blocked_bootstrap(*objects, blocks, n_iterations):
    """
    Repeatedly bootstrap the provided arrays across the specified dimension(s) and
    stack the new arrays along a new "iteration" dimension. The boostrapping is
    done in a nested manner. I.e. bootstrap the first provided dimension, then for
    each bootstrapped sample along that dimenion, bootstrap the second provided
    dimension, then for each bootstrapped sample along that dimenion...

    Note, this function expands out the iteration dimension inside a universal
    function. However, this can generate very large chunks (it multiplies chunk size
    by the number of iterations) and it falls over for large numbers of iterations
    for reasons I don't understand. It is thus best to apply this function in blocks
    using `iterative_blocked_bootstrap`

    Parameters
    ----------
    objects : iterable of Datasets
        The data to bootstrap. Multiple datasets can be passed to be bootstrapped
        in the same way. Where multiple datasets are passed, all datasets need not
        contain all bootstrapped dimensions. However, because of the bootstrapping
        is applied in a nested manner, the dimensions in all input objects must also
        be nested. E.g., for `dim=['d1','d2','d3']` an object with dimensions 'd1'
        and 'd2' is valid but an object with only dimension 'd2' is not.
    blocks : dict
        Dictionary of the dimension(s) to bootstrap and the block sizes to use along
        each dimension: {dim: blocksize}.
    n_iterations : int
        The number of times to repeat the bootstrapping
    """

    def _bootstrap(*arrays, indices):
        """Bootstrap the array(s) using the provided indices"""
        bootstrapped = [array[ind] for array, ind in zip(arrays, indices)]
        if len(bootstrapped) == 1:
            return bootstrapped[0]
        else:
            return tuple(bootstrapped)

    dim = list(blocks.keys())
    if isinstance(dim, str):
        dim = [dim]

    # Get the sizes of the bootstrap dimensions
    sizes = None
    for obj in objects:
        try:
            sizes = OrderedDict({d: (obj.sizes[d], b) for d, b in blocks.items()})
            break
        except KeyError:
            pass
    if sizes is None:
        raise ValueError("At least one input object must contain all dimensions in dim")

    # Generate the random indices first so that we can be sure that each dask chunk
    # uses the same indices. Note, I tried using random.seed() to achieve this but it
    # was flaky. These are the indices to bootstrap all objects.
    nested_indices = _n_nested_blocked_random_indices(sizes, n_iterations)

    # Need to expand the indices for broadcasting for each object separately
    # as each object may have different dimensions
    indices = []
    input_core_dims = []
    for obj in objects:
        available_dims = [d for d in dim if d in obj.dims]
        indices_to_expand = [nested_indices[key] for key in available_dims]

        # Check that dimensions are nested
        ndims = [i.ndim for i in indices_to_expand]
        if ndims != list(range(2, len(ndims) + 2)):  # Start at 2 due to iteration dim
            raise ValueError("The dimensions of all inputs must be nested")

        indices.append(_expand_n_nested_random_indices(indices_to_expand))
        input_core_dims.append(available_dims)

    # Loop over objects because they may have non-matching dimensions and
    # we don't want to broadcast them as this will unnecessarily increase
    # chunk size for dask arrays
    result = []
    for obj, ind, core_dims in zip(objects, indices, input_core_dims):
        # Assume all variables have the same dtype
        output_dtype = obj[list(obj.data_vars)[0]].dtype

        result.append(
            xr.apply_ufunc(
                _bootstrap,
                obj,
                kwargs=dict(
                    indices=[ind],
                ),
                input_core_dims=[core_dims],
                output_core_dims=[core_dims + ["iteration"]],
                dask="parallelized",
                dask_gufunc_kwargs=dict(output_sizes={"iteration": n_iterations}),
                output_dtypes=[output_dtype],
            )
        )

    return tuple(result)


def iterative_blocked_bootstrap(*objects, blocks, n_iterations):
    """
    Repeatedly bootstrap the provided arrays across the specified dimension(s) and
    stack the new arrays along a new "iteration" dimension. The boostrapping is
    done in a nested manner. I.e. bootstrap the first provided dimension, then for
    each bootstrapped sample along that dimenion, bootstrap the second provided
    dimension, then for each bootstrapped sample along that dimenion...

    Parameters
    ----------
    objects : iterable of Datasets
        The data to bootstrap. Multiple datasets can be passed to be bootstrapped
        in the same way. Where multiple datasets are passed, all datasets need not
        contain all bootstrapped dimensions. However, because of the bootstrapping
        is applied in a nested manner, the dimensions in all input objects must also
        be nested. E.g., for `dim=['d1','d2','d3']` an object with dimensions 'd1'
        and 'd2' is valid but an object with only dimension 'd2' is not.
    blocks : dict
        Dictionary of the dimension(s) to bootstrap and the block sizes to use along
        each dimension: {dim: blocksize}.
    n_iterations : int
        The number of times to repeat the bootstrapping
    """
    # The fastest way to perform the iterations is to expand out the iteration
    # dimension inside the universal function (see _iterative_bootstrap).
    # However, this can generate very large chunks (it multiplies chunk size by
    # the number of iterations) and it falls over for large numbers of iterations
    # for reasons I don't understand. Thus here we loop over blocks of iterations
    # to generate the total number of iterations.

    # Choose iteration blocks to limit chunk size on dask arrays
    if objects[
        0
    ].chunks:  # TO DO: this is not a very good check that input is dask array
        MAX_CHUNK_SIZE_MB = 200
        ds_max_chunk_size_MB = max([utils.max_chunk_size_MB(obj) for obj in objects])
        blocksize = int(MAX_CHUNK_SIZE_MB / ds_max_chunk_size_MB)
        if blocksize > n_iterations:
            blocksize = n_iterations
        if blocksize < 1:
            blocksize = 1
    else:
        blocksize = n_iterations

    bootstraps = []
    for _ in range(blocksize, n_iterations + 1, blocksize):
        bootstraps.append(
            _iterative_blocked_bootstrap(
                *objects, blocks=blocks, n_iterations=blocksize
            )
        )

    leftover = n_iterations % blocksize
    if leftover:
        bootstraps.append(
            _iterative_blocked_bootstrap(*objects, blocks=blocks, n_iterations=leftover)
        )

    return tuple(
        [
            xr.concat(b, dim="iteration", coords="minimal", compat="override")
            for b in zip(*bootstraps)
        ]
    )


# Skill score calculation
# ===============================================


def _calculate_metric_from_timeseries(
    *timeseries,
    metric,
    metric_kwargs,
    significance=True,
    transform=None,
    alpha=0.1,
):
    """
    Calculate a skill metric from the provided timeseries

    Statistical significance at 1-alpha is
    identified at all points where the sample skill metric is positive (negative) and
    the fraction of transformed values in the bootstrapped distribution below (above)
    no_skill_value--defining the p-values--is less than or equal to alpha.)
    """
    skill = metric(*timeseries, **metric_kwargs)

    if significance:
        bootstrapped_skill = metric(
            *iterative_blocked_bootstrap(
                *timeseries,
                blocks={"time": 5, "member": 1},
                n_iterations=N_BOOTSTRAP_ITERATIONS,
            ),
            **metric_kwargs,
        )

        no_skill = 0
        sample_skill = skill.copy()
        if transform:
            no_skill = transform(no_skill)
            sample_skill = transform(sample_skill)
            bootstrapped_skill = transform(bootstrapped_skill)

        pos_signif = (
            xr.where(bootstrapped_skill < no_skill, 1, 0).mean("iteration") <= alpha
        ) & (sample_skill > no_skill)
        neg_signif = (
            xr.where(bootstrapped_skill > no_skill, 1, 0).mean("iteration") <= alpha
        ) & (sample_skill < no_skill)

        significance = pos_signif | neg_signif
        significance = significance.rename(
            {n: f"{n}_signif" for n in significance.data_vars}
        )
        skill = xr.merge((skill, significance))

    return skill


def calculate_metric(
    hindcast,
    observation,
    metric,
    reference=None,
    metric_kwargs={},
    significance=False,
    transform=None,
    alpha=0.05,
):
    """
    Calculate a skill metric for a set of hindcasts. This function will attempt
    to validate over a common set of verification times at all leads and will
    return a warning if this is not possible. If a common set of verification
    times cannot be found, this function will verify over all available times
    at each lead

    Parameters
    ----------
    hindcast : xarray Dataset
        The hindcast data to verify. Must have "init" and "lead" dimensions
    observation : xarray Dataset
        The data to verify against. Must have a "time" dimension
    metric : str
        The name of the metric to apply to apply to the timeseries. Will look
        for function in src.verify.
    reference : str or xarray Dataset, optional
        The skill baseline. Can be an xarray Dataset with a "time" dimension
        (e.g. an historical simulation) that will be passed on to the `metric`
        function. Alternatively, users can specify one of the following
        methods to automatically create the appropriate baseline timeseries
        to be passed on to the `metric` function:
        - "persistence": hindcasts skill is baselined against the most recent
          observation at the time of forecast initialisation
        - "climatology": hindcasts skill is baselined against the monthly
          climatological mean of the observations over the verification period
    metric_kwargs : dict
        kwargs to pass to the function `metric`
    significance : boolean, optional
        If True, also return a mask indicating points where skill estimates are
        significant using the non-parametric bootstrapping approach of Goddard
        et al. (2013).
    transform : function, optional
        Transform to apply prior to estimating significant points
    alpha : float, optional
        The level [0,1] to apply sigificance at. Statistical significance at
        1-alpha is identified at all points where the sample skill metric is
        positive (negative) and the fraction of transformed values in the
        bootstrapped distribution below (above) zero--defining the p-values
        --is less than or equal to alpha.)
    """

    def _generate_persistence_hindcast(hindcast, observation):
        """
        Generate a persistence hindcast using the most recent observation at
        the time of forecast initialisation
        """
        # Use the index increment between the times at lead 0 and lead 1 in the
        # observations to determine how much we need to shift the observations
        # to get the most recent observation at the time of initialisation
        lead_increment = np.diff(np.insert(hindcast.lead, 0, -1))
        assert all(
            lead_increment == lead_increment[0]
        ), "lead increment must be the same for all leads for persistence baseline"
        times_of_lead_0 = hindcast.time.isel(lead=0)
        lead_0_indices = np.searchsorted(observation.time, times_of_lead_0.values)
        lead_1_index = np.searchsorted(
            observation.time, hindcast.time.isel(lead=1, init=0).values
        )
        shift = lead_1_index - lead_0_indices[0]

        # Get the most recent observation at the time of forecast initialisation
        lead_0_spacing = np.diff(lead_0_indices)
        assert all(
            lead_0_spacing == lead_0_spacing[0]
        ), "time spacing along observation must be regular for persistence baseline"
        persistence_lead_0 = observation.isel(time=lead_0_indices - shift)

        # Replicate the most recent observations at all leads
        persistence = (
            persistence_lead_0.rename({"time": "init"})
            .assign_coords({"init": hindcast.init})
            .broadcast_like(hindcast, exclude=["member"])
            .assign_coords({"time": hindcast.time})
            .assign_coords({PERSIST_TIME_DIM: ("init", persistence_lead_0.time.values)})
        )
        return persistence

    def _generate_climatology_timeseries(observation):
        """
        Generate a timeseries of monthly climatological values
        """
        clim = observation.groupby("time.month").mean("time")
        return (xr.zeros_like(observation).groupby("time.month") + clim).drop("month")

    def _common_set_of_verif_times(hcst, *refs, search_dim="lead"):
        """
        Get the common set of verification times available at all leads

        Stolen and adapted from climpred.alignment._same_verifs_alignment
        """
        hcst_times = hcst.time.compute()
        if len(refs) > 1:
            valid_times = xr.align(*[ref.time for ref in refs])[0].values
        else:
            valid_times = refs[0].time.values
        times = [
            i for i in valid_times if (i == hcst_times).any("init").all(search_dim)
        ]
        return times

    def _reindex_hindcast(hindcast):
        """
        Reindex hindcast dataset that is indexed by initial date and lead time
        to be indexed by target date and lead time
        """
        result = []
        for lead in hindcast["lead"]:
            hcst = hindcast.sel({"lead": lead}).swap_dims({"init": "time"}).drop("init")
            result.append(hcst)
        return xr.concat(result, dim="lead")

    TIME_INDEX_DIM = "i"
    VERIF_TIME_DIM = "verification_time"
    PERSIST_TIME_DIM = "persistence_time"

    logger = logging.getLogger(__name__)

    persistence_reference = False
    climatology_reference = False
    if isinstance(reference, xr.Dataset):
        assert "time" in reference.dims, "Reference Dataset must have a time dimension"
        references = [observation, reference]
    else:
        if isinstance(reference, str) & (reference == "persistence"):
            persistence_reference = True
        elif isinstance(reference, str) & (reference == "climatology"):
            climatology_reference = True
        elif reference is not None:
            raise ValueError("Unrecognised input for `reference`")
        references = [observation]

    verif_times = _common_set_of_verif_times(hindcast, *references)

    if persistence_reference:
        persistence_hindcast = _generate_persistence_hindcast(hindcast, observation)

    # Look for metric and transform in this module
    metric = getattr(sys.modules[__name__], metric)
    if transform is not None:
        transform = getattr(sys.modules[__name__], transform)

    if len(verif_times) > 0:
        references_verif_times = [r.sel(time=verif_times) for r in references]
        hindcast_verif_times = _reindex_hindcast(hindcast).sel(time=verif_times)

        if persistence_reference:
            persistence_verif_times = _reindex_hindcast(persistence_hindcast).sel(
                time=verif_times
            )
            references_verif_times.append(persistence_verif_times)

        if climatology_reference:
            observation_verif_times = references_verif_times[0]
            climatology_verif_times = _generate_climatology_timeseries(
                observation_verif_times
            )
            references_verif_times.append(climatology_verif_times)

        logger.info(
            (
                f"Performing verification over {verif_times[0].strftime('%Y-%m-%d')} "
                f"- {verif_times[-1].strftime('%Y-%m-%d')} using a common set of "
                f"verification times at all lead"
            )
        )

        skill = _calculate_metric_from_timeseries(
            hindcast_verif_times,
            *references_verif_times,
            metric=metric,
            metric_kwargs=metric_kwargs,
            significance=significance,
            transform=transform,
            alpha=alpha,
        )

        # Add verification time information into dataset
        skill = skill.assign_coords(
            {
                VERIF_TIME_DIM: xr.DataArray(
                    verif_times, coords={TIME_INDEX_DIM: range(len(verif_times))}
                )
            }
        )
        if persistence_reference:
            persistence_time = (
                persistence_verif_times[PERSIST_TIME_DIM]
                .assign_coords({"time": range(len(verif_times))})
                .rename({"time": TIME_INDEX_DIM})
            )
            skill = skill.assign_coords({PERSIST_TIME_DIM: persistence_time})
    else:
        # If can't find common verif times, validate over all available times at each lead
        warnings.warn(
            (
                "A common set of verification times at all leads could not be found. "
                "Verifying over all available times at each lead"
            )
        )
        skill = []
        for lead in hindcast.lead:
            hindcast_at_lead = hindcast.sel(lead=lead).dropna(dim="init", how="all")

            verif_times = _common_set_of_verif_times(
                hindcast_at_lead, *references, search_dim=None
            )
            hindcast_verif_times = hindcast_at_lead.swap_dims({"init": "time"}).sel(
                time=verif_times
            )
            references_verif_times = [r.sel(time=verif_times) for r in references]

            if persistence_reference:
                persistence_verif_times = (
                    persistence_hindcast.sel(lead=lead)
                    .swap_dims({"init": "time"})
                    .sel(time=verif_times)
                    .drop("init")
                )
                references_verif_times.append(persistence_verif_times)

            if climatology_reference:
                observation_verif_times = references_verif_times[0]
                climatology_verif_times = _generate_climatology_timeseries(
                    observation_verif_times
                )
                references_verif_times.append(climatology_verif_times)

            skill_at_lead = _calculate_metric_from_timeseries(
                hindcast_verif_times,
                *references_verif_times,
                metric=metric,
                metric_kwargs=metric_kwargs,
                significance=significance,
                transform=transform,
                alpha=alpha,
            )

            # Add verification time information into dataset
            skill_at_lead = skill_at_lead.assign_coords(
                {
                    VERIF_TIME_DIM: xr.DataArray(
                        verif_times, coords={TIME_INDEX_DIM: range(len(verif_times))}
                    )
                }
            )
            if persistence_reference:
                persistence_time = persistence_verif_times[PERSIST_TIME_DIM].values
                skill_at_lead = skill_at_lead.assign_coords(
                    {PERSIST_TIME_DIM: (TIME_INDEX_DIM, persistence_time)}
                )
            skill.append(skill_at_lead)
        skill = xr.concat(skill, dim="lead")

    return skill


# Command line interface
# ===============================================


def verify(config, save_dir, save=True):
    """
    Prepare a skill metric according to a provided config file

    Parameters
    ----------
    config : str
        The name of the config file
    save_dir : str
        The directory to save to
    save : boolean, optional
        If True (default), save the prepared dataset(s) in zarr format to save_dir. If
        False, return an xarray Dataset containing the prepared data. The latter is
        useful for debugging
    """
    logger = logging.getLogger(__name__)

    cfg = utils.load_config(config)

    if "prepare" in cfg:
        prepared = []

        for identifier, params in cfg["prepare"].items():
            if "apply" in params:
                if "all" in params["apply"]:
                    params["apply"]["hindcast"] = params["apply"]["all"]
                    params["apply"]["observation"] = params["apply"]["all"]
                    if "reference" in params:
                        if params["reference"] not in ["persistence", "climatology"]:
                            params["apply"]["reference"] = params["apply"]["all"]
            else:
                params["apply"] = []

            hindcast = xr.open_zarr(
                f"{DATA_DIR}/{params['hindcast']}.zarr",
                decode_timedelta=False,
            ).unify_chunks()
            if "hindcast" in params["apply"]:
                hindcast = utils.composite_function(params["apply"]["hindcast"])(
                    hindcast
                )

            observation = xr.open_zarr(
                f"{DATA_DIR}/{params['observation']}.zarr",
                decode_timedelta=False,
            ).unify_chunks()
            if "observation" in params["apply"]:
                observation = utils.composite_function(params["apply"]["observation"])(
                    observation
                )

            if "reference" in params:
                if params["reference"] in ["persistence", "climatology"]:
                    reference = params["reference"]
                else:
                    reference = xr.open_zarr(
                        f"{DATA_DIR}/{params['reference']}.zarr",
                        decode_timedelta=False,
                    ).unify_chunks()

                    if "reference" in params["apply"]:
                        reference = utils.composite_function(
                            params["apply"]["reference"]
                        )(reference)
            else:
                reference = None

            logger.info(f"Processing {identifier}")
            ds = calculate_metric(
                hindcast, observation, reference=reference, **params["verify"]
            )

            prepared.append(ds)
            if save:
                ds = ds.chunk({}).unify_chunks()
                for var in ds.variables:
                    ds[var].encoding = {}
                ds.to_zarr(f"{save_dir}/{identifier}.zarr", mode="w")

        logger.info(f"Succeeded calculating all verification metrics")
        return prepared
    else:
        raise ValueError(f"No skill metrics were specified to prepare")


def main(config, config_dir, save_dir):
    """
    Prepare skill metrics according to a provided config file

    Parameters
    ----------
    config : str
        The name of the config file
    config_dir : str
        The directory containing the config file
    save_dir : str
        The directory to save to
    """
    logger = logging.getLogger(__name__)

    logger.info("Spinning up a dask cluster")
    local_directory = tempfile.TemporaryDirectory()
    with Client(processes=False, local_directory=local_directory.name) as client:
        logger.info(f"Preparing skill metrics using {config}")
        verify(f"{config_dir}/{config}", save_dir)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser(
        description="Prepare skill metrics according to a provided config file"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Configuration file to process",
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default=f"{PROJECT_DIR}/config/verify",
        help="Location of directory containing config file(s) to use, defaults to <project_dir>/config/verify",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=f"{PROJECT_DIR}/data/skill/",
        help="Location of directory to save skill data to, defaults to <project_dir>/data/skill/",
    )

    args = parser.parse_args()
    config = args.config
    config_dir = args.config_dir
    save_dir = args.save_dir

    main(config, config_dir, save_dir)
