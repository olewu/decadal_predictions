import xarray as xr
import numpy as np
import re

from decadal_predictions.config import data_paths, var_name_map

def load_verification(
        variable:str, 
        period:list,
        vf_type:str='ERA5',
    ) -> xr.Dataset:

    data_path = data_paths['verification'][vf_type]/variable

    vf_files = sorted(list(data_path.glob('*.nc')))
    
    # filter to relevant years:
    vf_files_filtered = [file for file in vf_files if period[0] <= int(file.stem.split('_')[-2]) <= period[-1]]
    
    # open dataset
    vf_ds = xr.open_mfdataset(vf_files_filtered)

    return vf_ds, vf_ds.coords

def get_hindcast_sequence(
        variable:str,
        period:str,
        lead_year_range:list,
        mod_type:str='NorCPM1',
        interpolate_coords:dict=None,
        interpolate_names = {'lon':'longitude','lat':'latitude'},
        N_deg:int=None,
        longitude_name:str='lon',
        latitude_name:str='lat',
        boundary:str='trim',
    ) -> xr.Dataset:

    """
    loads and pre-processes a set of hindcasts in a given period
    pre-processing contains coarsening the forecast resolution
    and constructing N-year running averages out of monthly files
    from first lead year
    """

    N = lead_year_range[-1] - lead_year_range[0] + 1

    # find all matching files:
    all_var_files = sorted(
        list(
            data_paths['hindcast'][mod_type].glob(
                '*/{0}_Amon_{1}_dcppA-hindcast_s*.nc'.format(var_name_map['long_to_cmip'][variable],mod_type)
            )
        )
    )

    # find all init years
    init_years = sorted(
        list(
            set(
                [re.search('_s(\d{4})-r',file.stem).groups()[0] for file in all_var_files if period[0] <= int(re.search('_s(\d{4})-r',file.stem).groups()[0]) <= period[-1]]
            )
        )
    )

    ds_lead_time_hc_ens = []

    # loop through initialization years:
    llt_year = [] # central lead time year (valid year)
    for iyear in init_years:

        iy = int(iyear)
        print(iyear)

        # find all files for initialization:
        all_files_for_i = sorted([fi for fi in all_var_files if f'_s{iyear}-r' in str(fi.stem)])

        # load all hindcast members for init year
        ds_monthly_hc = xr.open_mfdataset(all_files_for_i,combine='nested', concat_dim='mem', preprocess=preprocess)
        
        if interpolate_coords is not None:
            ds_monthly_hc = ds_monthly_hc.interp(
                {
                    longitude_name:interpolate_coords[interpolate_names['lon']].values,
                    latitude_name:interpolate_coords[interpolate_names['lat']].values
                }
            )

        # smooth to desired res (first interpolate to original ERA5 grid and then coarsen with same settings):
        ds_monthly_hc_smooth = ds_downsample_to_N_degrees(
            ds_monthly_hc,N_deg=N_deg,longitude_name=longitude_name,latitude_name=latitude_name,boundary=boundary
        )

        # average annually:
        ds_annual_hc = ds_monthly_hc_smooth.groupby('time.year').mean()

        # average over lead time aggregation period:
        # year 2 defined as first full calendar year in (re-)forecast
        ys = iy + lead_year_range[0] - 2
        if ds_monthly_hc.time[0].dt.month > 1:
            ys += 1
        
        llt_year.append(ys+N-1) # last valid year

        # absoulte model values Y_{j,t} where j=iy and t the lead time period depending on N
        ds_lead_time_hc_ens.append(ds_annual_hc.sel(year=slice(ys,ys+(N-1))).mean('year'))
        
    # N-year running averages of var for chosen lead times:
    return xr.concat(ds_lead_time_hc_ens,xr.DataArray(data=llt_year,dims=['year'])).compute() # valid for one lead time (period)


def ds_downsample_to_N_degrees(
        ds:xr.Dataset,
        N_deg:int,
        longitude_name:str='longitude',
        latitude_name:str='latitude',
        boundary:str='trim'
    ) -> xr.Dataset:

    lon_diff = abs(ds[longitude_name].diff(dim=longitude_name).mean())
    lat_diff = abs(ds[latitude_name].diff(dim=latitude_name).mean())
    win_lon = int(N_deg//lon_diff)
    win_lat = int(N_deg//lat_diff)

    return ds.coarsen({longitude_name:win_lon}).mean().coarsen({latitude_name:win_lat},boundary=boundary).mean()

def running_Nyear_mean(
        ds:xr.Dataset,
        N:int,
        time_coord:str='time'  
    ) -> xr.Dataset:

    ds_annual = ds.groupby(f'{time_coord}.year').mean()
    return ds_annual.rolling(year=N).mean().dropna('year')


def loo_mean(vec) -> np.array:
    """
    compute the leave-one-out mean of a vector
    """
    total_sum = np.sum(vec)
    return (total_sum - vec) / (len(vec) - 1)

def loo_rolling_mean(vec,window_length:int=1) -> np.array:
    """
    compute the leave-k-out mean of a vector, where k is controlled by window_length
    returns a vector of same length as vec
    if window_length=0, the overall mean will be on every index
    """
    N = len(vec)
    result = np.zeros(N)
    
    for n in range(N):
        if n + window_length < N:  # Exclude range n to n+k
            included_values = np.concatenate((vec[:n], vec[n+window_length:]))
        else:  # At the end of the vector, include all values
            included_values = vec[:n]
        
        # Compute the mean of the included values
        result[n] = np.mean(included_values)
    
    return result

def preprocess(ds):
    ds = ds.copy()  # Avoid modifying the original dataset
    ds['lat'] = ds['lat'].round(6)
    ds['lon'] = ds['lon'].round(6)
    return ds

def make_season_group(ds:xr.Dataset,time_dim:str='time',months_in_season:list=[12, 1, 2, 3]) -> xr.DataArray:
    """
    
    """
    season = ds[f'{time_dim}.month'].isin(months_in_season)
    season = season.astype(int)
    season[season.astype(bool)] = ds.time.dt.year[season.astype(bool)]
    season[ds[f'{time_dim}.month'] == 12] += 1
    season.name = 'year'
    return season