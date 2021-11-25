import dask
import xarray as xr

def open_mfzarr(files, preprocess=None, parallel=False, **open_kwargs):
    """ Open multiple zarr files applying a preprocess step prior to merging 
        
        NOTE: for some reason, using parallel=True produces cftime.datetime objects
        that break with DatatimeAccessor methods like dt.floor. Opening the data in 
        serial does not, see https://github.com/pydata/xarray/issues/6026
    """
    open_zarr_ = dask.delayed(xr.open_zarr) if parallel else xr.open_zarr
    preprocess_ = dask.delayed(preprocess) if parallel else preprocess
    open_tasks = [open_zarr_(f, **open_kwargs) for f in files]
    preprocess_tasks = [
        preprocess_(task) for task in open_tasks] if preprocess is not None else open_tasks
    datasets = dask.compute(preprocess_tasks)[0] if parallel else preprocess_tasks
    return xr.combine_by_coords(
        datasets, compat='override', coords='minimal').unify_chunks()