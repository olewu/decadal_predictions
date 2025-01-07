import xarray as xr

from decadal_predictions.config import *
from decadal_predictions.utils import *

def compute_ref_winter_nao(
        ref_type:str,
        period:list,
        ref_period:list=[1960,2015],
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

    # filter to winter averages (DJFM):
    season = make_season_group(ds,months_in_season=NAO_months)
    northern_seas = northern_center.groupby(season).mean(ds_coords['time_dim']).sel(year=slice(*ref_period))
    southern_seas = southern_center.groupby(season).mean(ds_coords['time_dim']).sel(year=slice(*ref_period))

    # compute annual winter NAO index
    NAO_raw = southern_seas - northern_seas
    # normalize to 0 mean:
    # NAO_anom = NAO_raw - NAO_raw.mean()

    # calculate anomalies wrt a leave-k-year-out mean of the NAO for fairer 
    # comparison to a forecast situation where year under consideration is unavailble:
    NAO_lko_anom = NAO_raw - lko_mean(NAO_raw[var_name_map['long_to_cf'][var]],N_run)

    # compute rolling average (time tag refers to last year in the window)
    # NAO_anom_8y_run = NAO_anom.rolling(year=N_run).mean().dropna('year')
    NAO_lko_anom_8y_run = NAO_lko_anom.rolling(year=N_run).mean().dropna('year')
    
    return NAO_lko_anom_8y_run

if __name__=='__main__':
    
    # verfication:
    ref_type = 'ERA5'
    period = [1960,2023] # period to load data for
    ref_period = [1960,2018] # anomalies are computed relative to the average over this period
    # for N_run in [4,8]:
    #     NAO_anom_vf = compute_ref_winter_nao(ref_type,period,ref_period=ref_period,N_run=8) # msl in Pa
    #     vf_idx_path = data_paths['processed']/f'verification/{ref_type}/indexes'
    #     vf_idx_path.mkdir(exist_ok=True,parents=True)
    #     NAO_anom_vf.to_netcdf(
    #         vf_idx_path/'NAO_stat_{1}-{2}_{0}yrm.nc'.format(
    #             N_run,ref_period[0],ref_period[-1]
    #         )
    #     )

    lead_year_ranges = [[2,5],[6,9],[2,9]]

    # hindcasts:
    for mod in available_models:
        hc_type = model_name_map[mod]
        for lead_year_range in lead_year_ranges:
            print(hc_type,lead_year_range)
            NAO_hc = get_hindcast_sequence(
                'mean_sea_level_pressure',
                ref_period,
                lead_year_range=lead_year_range,
                mod_type=hc_type,
                process='station_NAO',
                mem_dim='mme_member'
            ) # msl in Pa

            NAO_anom_hc = NAO_hc - NAO_hc.mean('year')
            # save absolute unnormalized NAO index:
            hc_idx_path = data_paths['processed']/f'hindcast/{hc_type}/indexes'
            hc_idx_path.mkdir(parents=True,exist_ok=True)
            NAO_anom_hc.to_netcdf(
                hc_idx_path/'NAO_stat_{0}-{1}_LY{2}-{3}.nc'.format(
                    ref_period[0],ref_period[-1],lead_year_range[0],lead_year_range[-1]
                )
            )


    # NAO_anom_hc.psl.plot(hue='realization',color='lightgrey')
    # NAO_anom_hc.mean('realization').psl.plot(color='C0')
    # NAO_anom_vf.msl.plot(color='C1')

    # xr.corr(NAO_anom_hc.mean('realization').psl,NAO_anom_vf.msl)