import xarray as xr
import numpy as np
import re
# from functools import partial
from pathlib import Path

from decadal_predictions.config import data_paths, var_name_map, NAO_box, model_encoding, model_decoding, example_grid_file

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

    return vf_ds

def get_hindcast_sequence(
        variable:str,
        period:str,
        lead_year_range:list,
        mod_type:str='NorCPM1',
        interpolate_coords:dict=None,
        interpolate_names:dict = {'lon':'longitude','lat':'latitude'},
        process=None,
        N_deg:int=None,
        longitude_name:str='lon',
        latitude_name:str='lat',
        boundary:str='trim',
        mem_dim:str='mme_member'
    ) -> xr.Dataset:

    """
    loads and pre-processes a set of hindcasts in a given period
    pre-processing contains coarsening the forecast resolution
    and constructing N-year running averages out of monthly files
    from first lead year
    """

    N = lead_year_range[-1] - lead_year_range[0] + 1

    # find all matching files:
    search_path = Path(str(data_paths['hindcast'][mod_type])+'_rgr')
    
    all_var_files = sorted(
        list(
            search_path.rglob(
                '{0}_Amon_{1}_dcppA-hindcast_s*.nc'.format(var_name_map['long_to_cmip'][variable],mod_type)
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
        if len(all_files_for_i) == 1:
            ds_monthly_hc = xr.open_dataset(all_files_for_i[0])
            print('WARN: need to implement changing coordinate to mme_member encoding!')
            #TODO: change the 'realization' coordinate to mme_member encoding!
        else:
            ds_monthly_hc = xr.open_mfdataset(all_files_for_i,combine='nested', concat_dim='mme_member', preprocess=preprocess_ensdim)

        if (mem_dim != 'mme_member') & (mem_dim in list(ds_monthly_hc.coords)):
            ds_monthly_hc = ds_monthly_hc.rename({mem_dim:'mme_member'})

        if interpolate_coords is not None:
            ds_monthly_hc = ds_monthly_hc.interp(
                {
                    longitude_name:interpolate_coords[interpolate_names['lon']].values,
                    latitude_name:interpolate_coords[interpolate_names['lat']].values
                }
            )

        if N_deg is not None:
            # smooth to desired res (first interpolate to original ERA5 grid and then coarsen with same settings):
            ds_monthly_hc = ds_downsample_to_N_degrees(
                ds_monthly_hc,N_deg=N_deg,longitude_name=longitude_name,latitude_name=latitude_name,boundary=boundary
            )

        if process == 'station_NAO':
            ds_processed_hc = compute_hindcast_nao(ds_monthly_hc)
        else:
            # average annually:
            ds_processed_hc = ds_monthly_hc.groupby('time.year').mean()

        # average over lead time aggregation period:
        # year 2 defined as first full calendar year in (re-)forecast
        ys = iy + lead_year_range[0] - 2
        if ds_monthly_hc.time[0].dt.month > 1:
            ys += 1
        
        llt_year.append(ys+N-1) # last valid year

        # absoulte model values Y_{j,t} where j=iy and t the lead time period depending on N
        ds_lead_time_hc_ens.append(ds_processed_hc.sel(year=slice(ys,ys+(N-1))).mean('year').compute())
        
    # N-year running averages of var for chosen lead times:
    return xr.concat(ds_lead_time_hc_ens,xr.DataArray(data=llt_year,dims=['year'])) # valid for one lead time (period)


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

def adjust_latlon(ds,origin_grid_file=example_grid_file):

    ds_grd = xr.open_dataset(origin_grid_file)
    
    ds_adj = ds.copy()

    # adjust the longitude coordinate:
    ds_adj.coords['longitude'] = (ds_adj.coords['longitude'] + 180) % 360 - 180
    ds_adj = ds_adj.sortby(ds_adj.longitude)

    # interpolate:
    ds_adj = ds_adj.interp(latitude=ds_grd.latitude,longitude=ds_grd.longitude)

    ds_adj = ds_adj.reindex(latitude=list(reversed(ds_adj.latitude)))
    
    return ds_adj


def compute_hindcast_nao(ds_monthly):
    """
    this computes the NAO value for a single (!) hindcast and is implemented in
    utils.load_hindcast_sequence() as processor, so to get an NAO sequence,
    utils.load_hindcast_sequence() needs to be called with option NAO
    takes monthly dataset as input and returns annual values of the NAO,
    the rolling average computation is done in utils.load_hindcast_sequence()
    """

    ds_coords = find_coordnames(ds_monthly)

    iyear = int(ds_monthly['{}.year'.format(ds_coords['time_dim'])].values[0])

    # compute NAO:
    northern_center = ds_monthly.sel(
        {ds_coords['lat_dim']:NAO_box['north']['latitude'],ds_coords['lon_dim']:NAO_box['north']['lon']}
    ).mean([ds_coords['lat_dim'],ds_coords['lon_dim']])
    southern_center = ds_monthly.sel(
        {ds_coords['lat_dim']:NAO_box['south']['latitude'],ds_coords['lon_dim']:NAO_box['south']['lon']}
    ).mean([ds_coords['lat_dim'],ds_coords['lon_dim']])

    season = make_season_group(ds_monthly)

    northern_seas = northern_center.groupby(season).mean(ds_coords['time_dim']).sel(year=slice(iyear,iyear+10))
    southern_seas = southern_center.groupby(season).mean(ds_coords['time_dim']).sel(year=slice(iyear,iyear+10))

    NAO_raw = southern_seas - northern_seas #).compute()

    return NAO_raw


def loo_mean(vec) -> np.array:
    """
    compute the leave-one-out mean of a vector
    """
    total_sum = np.sum(vec)
    return (total_sum - vec) / (len(vec) - 1)

def find_coordnames(ds):
    ds_coords = ds.coords
    # find lon lat:
    lat_dim = [co for co in list(ds_coords) if 'lat' in co][0]
    lon_dim = [co for co in list(ds_coords) if 'lon' in co][0]
    time_dim = [co for co in list(ds_coords) if co in ['time','date']][0]

    return {'lat_dim':lat_dim,'lon_dim':lon_dim,'time_dim':time_dim}

def lko_mean(vec,k_window_length:int=1) -> np.array:
    """
    compute the leave-k-out mean of a vector, where k is controlled by window_length
    returns a vector of same length as vec
    if window_length=0, the overall mean will be on every index
    """
    N = len(vec)
    result = np.zeros(N)
    
    for n in range(N):
        if n + k_window_length < N:  # Exclude range n to n+k
            included_values = np.concatenate((vec[:n], vec[n+k_window_length:]))
        else:  # At the end of the vector, include all values
            included_values = vec[:n]
        
        # Compute the mean of the included values
        result[n] = np.nanmean(included_values)
    
    return result

def lko_std(vec,k_window_length:int=1) -> np.array:
    """
    compute the leave-k-out mean of a vector, where k is controlled by window_length
    returns a vector of same length as vec
    if window_length=0, the overall mean will be on every index
    """
    N = len(vec)
    result = np.zeros(N)
    
    for n in range(N):
        if n + k_window_length < N:  # Exclude range n to n+k
            included_values = np.concatenate((vec[:n], vec[n+k_window_length:]))
        else:  # At the end of the vector, include all values
            included_values = vec[:n]
        
        # Compute the mean of the included values
        result[n] = np.nanstd(included_values)
    
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


def encode_model_ensemble_from_file_name(model,variant):
    # model encoding
    mcode = model_encoding[model]
    # variant encoding
    rc,ic,pc,fc = re.search(r'r(\d{1,2})i(\d{1,2})p(\d{1,2})f(\d{1,2})',variant).groups()
    
    men_code = '{0:02}{1}{2}{3}{4}'.format(mcode,rc.zfill(2),ic.zfill(2),pc.zfill(2),fc.zfill(2))
    return int(men_code)

def model_ensemble_decoder(men_code):

    model = model_decoding(men_code[:2])

    variant = 'r{0}i{1}p{2}f{3}'.format(int(men_code[2:4]),int(men_code[4:6]),int(men_code[6:8]),int(men_code[8:10]))

    return model,variant

def preprocess_ensdim(ds):
    model = ds.attrs['source_id']
    variant = ds.attrs['variant_label']
    coord = encode_model_ensemble_from_file_name(model,variant)
    return ds.expand_dims({"mme_member": [coord]})



# from pathlib import Path
# HC_PATH = Path('/datalake/NS9873K/DATA/DCPP/dcppA-hindcast/')
# for pth in HC_PATH.rglob('*.nc'):
#     try:
#         iyear = re.search(r'_(\d{4})\d{4}-\d{8}.',str(pth.name)).groups()[0]
#         npth = Path(str(pth).replace('hindcast_r',f'hindcast_s{iyear}-r'))
#         pth.replace(npth)
#     except:
#         print(pth)
#         pass

# for pth in HC_PATH.rglob('psl*dcppA*.nc'):
#     repstr = re.search('_s\d{4}_r',pth.stem).group(0)
#     npth = str(pth).replace(repstr,repstr.replace('_r','-r'))
#     pth.replace(npth)

# for pth in HC_PATH.rglob('*_-r*nc'):
#     npth = str(pth).replace('_-r','-r')
#     pth.replace(npth)

# for pth in HC_PATH.rglob('*.nc'):
#     try:
#         repstr = re.search('_((s\d{4})\d{4})-r',pth.stem)
#         npth = str(pth).replace(repstr.group(0),repstr.group(0).replace(repstr.group(1),repstr.group(2)))
#         pth.replace(npth)
#     except:
#         pass    