import geopandas as gpd


# Mapping from the shared time_agg token to the pandas/xarray offset alias
_TIME_AGG_FREQ = {
    's':     '1s',
    'm':     '1min',
    '15min': '15min',
    '30min': '30min',
    'h':     'h',
}


def consumption_time_agg(da, time_agg='h'):
    """
    Resamples a consumption xarray DataArray to a coarser time resolution.

    Similar to ``discharge_time_agg`` helper in ``wastewater_quality`` but
    operates on xarray DataArrays rather than pandas DataFrames.

    Args:
        da (xarray.DataArray): Consumption DataArray with a ``time`` dimension.
            Typically produced by summing ``house.consumption`` over end-uses
            and users. Values are in L/s.
        time_agg (str): Target time resolution. Accepted values:

            * ``'s'``     — keep at seconds (no change)
            * ``'m'``     — 1-minute means
            * ``'15min'`` — 15-minute means
            * ``'30min'`` — 30-minute means
            * ``'h'``     — hourly means (default, matches ``hh_discharge_nutrients``)

    Returns:
        xarray.DataArray: Resampled DataArray at the requested resolution.

    Raises:
        ValueError: If ``time_agg`` is not one of the accepted values.
    """
    if time_agg not in _TIME_AGG_FREQ:
        raise ValueError(
            f"Invalid time_agg '{time_agg}'. "
            f"Use one of: {list(_TIME_AGG_FREQ.keys())}"
        )
    freq = _TIME_AGG_FREQ[time_agg]
    return da.resample(time=freq).mean()


def fix_invalid_geometries(gdf):
    """
    Fixes invalid geometries in a GeoDataFrame using the buffer(0) trick.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame to fix.

    Returns:
        gpd.GeoDataFrame: The GeoDataFrame with fixed geometries.
    """
    gdf['geometry'] = gdf['geometry'].buffer(0)
    
    return gdf