import xarray as xr

from decadal_predictions.config import *
from decadal_predictions.utils import load_verification, find_coordnames, make_season_group, ext_from_months, get_hindcast_sequence, lko_mean
from joblib import Parallel, delayed

def compute_ref_nao(
        ref_type:str,
        period:list,
        ref_period:list=[1961,2015],
        N_run:int=1,
        var:str='mean_sea_level_pressure',
        NAO_months:list=[12,1,2,3],
    ):
    """
    each NAO value is a N_run-year running average of the previous N_run year (including the tagged year)
    """
    
    ds = load_verification(var,period,ref_type)

    ds_coords = find_coordnames(ds)

    # compute SLP averages in centers of action (absolute monthly data):
    northern_center = ds.sel(
        {ds_coords['lat_dim']:NAO_box['north']['latitude'],ds_coords['lon_dim']:NAO_box['north']['longitude']}
    ).mean(
        [ds_coords['lat_dim'],ds_coords['lon_dim']]
    ).compute()
    
    southern_center = ds.sel(
        {ds_coords['lat_dim']:NAO_box['south']['latitude'],ds_coords['lon_dim']:NAO_box['south']['longitude']}
    ).mean(
        [ds_coords['lat_dim'],ds_coords['lon_dim']]
    ).compute()
    # since the areas are relatively small, area weighting has a negligible effect

    if NAO_months:
        # filter to winter averages (DJFM):
        season = make_season_group(ds,months_in_season=NAO_months)
        northern_seas = northern_center.groupby(season).mean(ds_coords['time_dim']).sel(year=slice(*ref_period))
        southern_seas = southern_center.groupby(season).mean(ds_coords['time_dim']).sel(year=slice(*ref_period))
    else:
        northern_seas = northern_center.groupby(ds_coords['time_dim']+'.year').mean(ds_coords['time_dim']).sel(year=slice(*ref_period))
        southern_seas = southern_center.groupby(ds_coords['time_dim']+'.year').mean(ds_coords['time_dim']).sel(year=slice(*ref_period))

    # compute annual winter NAO index
    NAO_raw = southern_seas - northern_seas
    # normalize to 0 mean:
    # NAO_anom = NAO_raw - NAO_raw.mean()

    # calculate anomalies wrt a leave-k-year-out mean of the NAO for fairer 
    # comparison to a forecast situation where year under consideration is unavailble:
    NAO_lko_anom = NAO_raw - lko_mean(NAO_raw[var_name_map['long_to_cf'][var]],N_run)
    # NAO_lko_anom = NAO_raw - NAO_raw[var_name_map['long_to_cf'][var]].mean(('year'))

    # compute rolling average (time tag refers to last year in the window)
    # NAO_anom_Ny_run = NAO_anom.rolling(year=N_run).mean().dropna('year')
    NAO_lko_anom_Ny_run = NAO_lko_anom.rolling(year=N_run).mean().dropna('year')
    
    return NAO_lko_anom_Ny_run

def compute_hindcast_sequence_nao(hc_type,lead_year_range,ref_period,hc_period,NAO_months):
    print(hc_type,lead_year_range)      
    NAO_hc = get_hindcast_sequence(
        'mean_sea_level_pressure',
        hc_period,
        lead_year_range=lead_year_range,
        mod_type=hc_type,
        process='station_NAO',
        months_in_season=NAO_months,
        mem_dim='mme_member'
    ) # msl in Pa

    N = lead_year_range[-1] - lead_year_range[0] + 1
    # leave N years out climatology of NAO:
    NAO_clim_lko = xr.apply_ufunc(lko_mean,NAO_hc.sel(year=slice(*ref_period)).mean('mme_member'),N,input_core_dims=[['year'],[]],output_core_dims=[['year',]],vectorize=True)
    # extrapolate the climatological value to the future values:
    NAO_clim_lko = NAO_clim_lko.interp(year=NAO_hc.year, method='nearest', kwargs={"fill_value": None}).ffill('year')
    NAO_clim_lko['iyear'] = NAO_hc['iyear']
    # NAO_clim_lko = NAO_hc.sel(year=slice(*ref_period)).mean(('year','mme_member'))
    NAO_anom_hc = NAO_hc - NAO_clim_lko
    # save absolute unnormalized NAO index:
    hc_idx_path = data_paths['processed']/f'hindcast/{hc_type}/indexes'
    hc_idx_path.mkdir(parents=True,exist_ok=True)
    seas_ext = ext_from_months(NAO_months)
    NAO_anom_hc.to_netcdf(
        hc_idx_path/'NAO_stat_{0}-{1}_LY{2}-{3}{4}.nc'.format(
            hc_period[0],hc_period[-1],lead_year_range[0],lead_year_range[-1],seas_ext
        )
    )

if __name__=='__main__':
    
    # verification:
    ref_type = 'ERA5'
    period = [1960,2023] # period to load data for
    ref_period = [1961,2023] # anomalies are computed relative to the average over this period
    NAO_months = [12,1,2,3] # months to average over
    seas_ext = ext_from_months(NAO_months)
    for N_run in [4,8]:
        NAO_anom_vf = compute_ref_nao(ref_type,period,ref_period=ref_period,N_run=N_run,NAO_months=NAO_months) # msl in Pa
        vf_idx_path = data_paths['processed']/f'verification/{ref_type}/indexes'
        vf_idx_path.mkdir(exist_ok=True,parents=True)
        NAO_anom_vf.to_netcdf(
            vf_idx_path/'NAO_stat{3}_{1}-{2}_{0}yrm.nc'.format(
                N_run,ref_period[0],ref_period[-1],seas_ext
            )
        )

    lead_year_ranges = [[2,5],[6,9],[2,9]]
    hc_period = [1960,2018] # refers to initialization dates
    njobs = len(available_models)*len(lead_year_ranges)
    # process hindcasts in parallel:
    prl = Parallel(n_jobs=njobs)(
        delayed(compute_hindcast_sequence_nao)(
            hc_type,lead_year_range,ref_period,hc_period,NAO_months=NAO_months
        ) for hc_type in available_models_long for lead_year_range in lead_year_ranges
    )
    
# from pathlib import Path          
# spth = Path('/projects/NS9873K/owul/data/statkraft_pilot4/decadal_predictions/hindcast')
# all_var = sorted(list(spth.rglob('*/indexes/NAO_stat_ann*.nc')))
# for varf in all_var:
#     newname = varf.name.replace('.nc','_ann.nc').replace('stat_ann','stat')
#     newvarf = varf.parent / Path(newname)
#     varf.replace(newvarf)