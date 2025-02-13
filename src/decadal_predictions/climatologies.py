import xarray as xr
from decadal_predictions.utils import *
from decadal_predictions.config import *
from joblib import Parallel, delayed

drop_vars = ['height','area','gw','lat_bnds','lon_bnds']

def hyb_climatology(ds_Nyear_rolling:xr.Dataset,clim_years:int=15):
    """
    compute a "hybrid-M" climatology as in Meehl et al. (2021, Clim Dyn, there: M=15), 
    which uses the M years prior to initialization as climatological reference
    to account for trends in the hindcast period and only uses the
    """

    hyb_clim = ds_Nyear_rolling.rolling(year=clim_years).mean().dropna('year') # this will contain the 15 years prior to the last year of the N year average
    
    # move year axis back by one year, such that the anomaly for year Y is computed wrt a 15yr +  N-1 years prior, excluding the year under consideration
    hyb_clim['year'] = hyb_clim['year'] + 1

    return hyb_clim

def standard_climatology(ds_Nyear_rolling:xr.Dataset,time_dim='year',leave_out_years_clim=1):
    """
    standard climatological mean over the entirety of the period in a leave-k-out sense
    """
    
    clim = xr.apply_ufunc(lko_mean,ds_Nyear_rolling,leave_out_years_clim,input_core_dims=[[time_dim],[]],output_core_dims=[[time_dim,]],vectorize=True)

    return clim



def make_vf_hybC_climatologies(vf_types,variables,Ns,period,smoothing_degrees,clim_years=15,save_clim=True,season=[],leave_out_years_clim=1):
    # collect coordinates of the verification to interpolate hindcasts to a matching grid later:
    # vf_coords = {}
    season_ext = ext_from_months(season)
    for vf_type in vf_types:
        # vf_coords[vf_type] = {}
        for var in variables:
            print(f'processing {vf_type} {var}')
            # load verfication dataset (monthly means)
            ds_vf = load_verification(var,period,vf_type=vf_type)
            ds_vf = adjust_latlon(ds_vf)
            # vf_coords[vf_type][var] = ds_vf.coords
            # coarsen resolution to required number of degrees
            ds_vf_coarse = ds_downsample_to_N_degrees(ds_vf,smoothing_degrees)
            if season:
                seas_grp = make_season_group(ds_vf_coarse,months_in_season=season)
                ds_vf_coarse = ds_vf_coarse.groupby(seas_grp).mean('time').sel(year=slice(period[0]-1,period[-1]+1))
            else:
                ds_vf_coarse = ds_vf_coarse.groupby('time.year').mean('time').sel(year=slice(period[0]-1,period[-1]+1))
            t_coord = 'year'
            # go through time aggregations:
            for N in Ns:
                # make running average:
                ds_vf_runmean = running_Nyear_mean(ds_vf_coarse,N,time_coord=t_coord).compute()
                if clim_years is None:
                    clim_type = f'standard_l{leave_out_years_clim}o'
                    ds_vf_clim = standard_climatology(ds_vf_runmean,time_dim=t_coord,leave_out_years_clim=leave_out_years_clim)
                else:
                    clim_type = f'hyb{clim_years}'
                    # compute the hybrid-15 climatology
                    ds_vf_clim = hyb_climatology(ds_vf_runmean,clim_years=clim_years)
                
                # compute anomalies and save:
                vf_anomalies = ds_vf_runmean - ds_vf_clim

                #------------filename------------#
                anom_path = data_paths['processed']/'verification/{0}/{1}'.format(
                    vf_type,var
                )
                # make path if it doesn't exist:
                anom_path.mkdir(parents=True,exist_ok=True)
                # build filename:
                anom_vf_filename = '{0}_{5}_anom_{1}-{2}_{3}yrm_{4}deg{6}.nc'.format(
                    var,period[0],period[-1],N,smoothing_degrees,clim_type,season_ext
                )
                
                #------------save------------#
                vf_anomalies.to_netcdf(anom_path/anom_vf_filename)
                
                if save_clim:
                    ds_vf_clim.to_netcdf(anom_path/anom_vf_filename.replace('_anom_','_clim_'))

    # return vf_coords

def make_hindcast_hybC_anomalies(hc_type,var,lead_year_range,hc_period,smoothing_degrees,ensmem_dim='mme_member',clim_years=15,save_clim=True,season=[],leave_out_years_clim=1):

    print(f'processing {hc_type} {var} {lead_year_range}')

    ds_hc_lt = get_hindcast_sequence(
        variable=var,
        period=hc_period,
        lead_year_range=lead_year_range,
        mod_type=hc_type,
        N_deg=smoothing_degrees,
        months_in_season=season,
    )
    if clim_years is None:
        clim_type = f'standard_l{leave_out_years_clim}o'
        hc_clim = standard_climatology(ds_hc_lt.mean(ensmem_dim),time_dim='year',leave_out_years_clim=1)
    else:
        clim_type = f'hyb{clim_years}'
        hc_clim = hyb_climatology(ds_hc_lt.mean(ensmem_dim),clim_years=clim_years)
    
    # drop unnecessary grid variables from the data
    drp_vrs = [drv for drv in drop_vars if drv in list(hc_clim)]
    hc_clim = hc_clim.drop_vars(drp_vrs)

    # compute anomalies:
    hc_anom = ds_hc_lt - hc_clim


    #------------filename------------#
    hc_anom_path = data_paths['processed']/'hindcast/{1}/{0}'.format(
        var,
        hc_type
    )
    hc_anom_path.mkdir(parents=True,exist_ok=True)
    season_ext = ext_from_months(season)
    hc_anom_filename = '{0}_{1}_anom_{2}-{3}_LY{4}-{5}_{6}deg{7}.nc'.format(
        var,clim_type,hc_period[0],hc_period[-1],lead_year_range[0],lead_year_range[-1],smoothing_degrees,season_ext
    )

    #------------save------------#
    hc_anom.to_netcdf(hc_anom_path/hc_anom_filename)
    if save_clim:
        hc_clim.to_netcdf(hc_anom_path/hc_anom_filename.replace('_anom_','_clim_'))

    return

def process_hindcast_climatologies(hc_types,variables,lead_year_ranges,hc_period,smoothing_degrees,clim_years=15,save_clim=True,season=[],leave_out_years_clim=1,ncpu=1):
    
    if ncpu == 1:
        for hc_type in hc_types:
            for var in variables:
                for lead_year_range in lead_year_ranges:
                    make_hindcast_hybC_anomalies(hc_type,var,lead_year_range,hc_period,smoothing_degrees,clim_years=clim_years,save_clim=save_clim,season=season,leave_out_years_clim=leave_out_years_clim)
    elif ncpu > 1:
        Parallel(n_jobs=ncpu)(delayed(make_hindcast_hybC_anomalies)(hc_type,var,lead_year_range,hc_period,smoothing_degrees,clim_years=clim_years,save_clim=save_clim,season=season,leave_out_years_clim=leave_out_years_clim) for hc_type in hc_types for var in variables for lead_year_range in lead_year_ranges)
    
    return

def make_MME_climatologies(variables,lead_year_ranges,hc_period,smoothing_degrees,NAO_match_members:int=None,ensmem_dim='mme_member',clim_years=15,save_clim=True,season_ext=None):
    
    for var in variables:
        for lead_year_range in lead_year_ranges:
            ly1 = lead_year_range[0]; lye = lead_year_range[-1]
            ds_hc_lt_mme = xr.open_dataset(data_paths['processed']/f'hindcast/MME/{var}/{var}_raw_1960-2018_LY{ly1}-{lye}_5deg{seas_ext}.nc')

            # subselect top ranked members for NAO if desired:
            if NAO_match_members is not None:
                # load NAO ranking:
                NAO_ranking = xr.open_dataarray(data_paths['processed']/f'hindcast/MME_ranking/NAO_match_ranking_MME_{ly1}-{lye}_ERA5_scaled.nc')
                ds_hc_lt = []
                for y in NAO_ranking['year']:
                    if y.values in ds_hc_lt_mme.year.values:
                        # print(y.values)
                        mme_sel = ds_hc_lt_mme.sel(year=y,mme_member=NAO_ranking.sel(year=y).isel(rank=slice(0,NAO_match_members)).values).expand_dims({'year':1})
                        mme_sel = mme_sel.rename({'mme_member':'rank'})
                        mme_sel['rank'] = np.arange(1,NAO_match_members+1)
                        ds_hc_lt.append(mme_sel)
                        # print(smallest_diff_index.sel(year=y).values)
                ds_hc_lt = xr.concat(ds_hc_lt,dim='year')
                mem_ext = '_top{0}'.format(NAO_match_members)
                ensmem_dim = 'rank'
            else:
                mem_ext = '_all'
                ds_hc_lt = ds_hc_lt_mme

            hc_clim = hyb_climatology(ds_hc_lt.mean(ensmem_dim),clim_years=clim_years)
            
            # drop unnecessary grid variables from the data
            drp_vrs = [drv for drv in drop_vars if drv in list(hc_clim)]
            hc_clim = hc_clim.drop_vars(drp_vrs)

            # compute anomalies:
            hc_anom = ds_hc_lt - hc_clim


            #------------filename------------#
            hc_anom_path = data_paths['processed']/'hindcast/MME/{0}'.format(var)
            hc_anom_path.mkdir(parents=True,exist_ok=True)
            hc_anom_filename = '{0}_hyb{1}_anom_{2}-{3}_LY{4}-{5}_{6}deg{7}{8}.nc'.format(
                var,clim_years,hc_period[0],hc_period[-1],lead_year_range[0],lead_year_range[-1],smoothing_degrees,mem_ext,season_ext
            )

            #------------save------------#
            hc_anom.to_netcdf(hc_anom_path/hc_anom_filename)
            if save_clim:
                hc_clim.to_netcdf(hc_anom_path/hc_anom_filename.replace('_anom_','_clim_'))

    return

if __name__ == '__main__':

    vf_types = ['ERA5']
    hc_types = [mod for mod in list(model_name_map.values())] # ['NorCPM1'] # ,
    variables = ['total_precipitation','2m_temperature','surface_net_solar_radiation','10m_wind_speed','mean_sea_level_pressure'] # , 
    Ns = [4,8]
    lead_year_ranges = [[2,9],[2,5],[6,9]]
    period = [1960,2023] # period to load data for
    hc_period = [1960,2018]
    smoothing_degrees = 5
    clim_years = None # choose None for computing standard (leave-one-year-out) climatology
    season = [] # choose [] to take full year

    # # compute verification climatologies: should take ~ 25s
    make_vf_hybC_climatologies(vf_types,variables,Ns,period,smoothing_degrees,clim_years,save_clim=True,season=season)

    # # this takes 2-3 mins for a single hindcast variable
    # # (single combination of (hc_type,var)) on NIRD login node, i.e. for all 5 variables ~ 3 mins * 5 = 15 mins and then times the number of models (6) ~ 6 * 15 mins = 90 mins
    process_hindcast_climatologies(hc_types,variables,lead_year_ranges,hc_period,smoothing_degrees,clim_years=clim_years,season=season,ncpu=30)

    # Make hindcast climatology for the MME:
    # make_MME_climatologies(variables,lead_year_ranges,hc_period,smoothing_degrees,clim_years=clim_years,season_ext=seas_ext)
    # Make hindcast climatology for the NAO-matched MME:
    # make_MME_climatologies(variables,lead_year_ranges,hc_period,smoothing_degrees,clim_years=clim_years,NAO_match_members=20,season_ext=seas_ext)

# spth = Path('/projects/NS9873K/owul/data/statkraft_pilot4/decadal_predictions/verification/ERA5/')
# hyb15_files = sorted(list(spth.rglob('*/*_hyb15*deg.nc')))
# for h15f in hyb15_files:
#     newname = h15f.name.replace('deg.nc','deg_ann.nc')
#     h15f.replace(h15f.parent/newname)