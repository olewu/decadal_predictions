import xarray as xr
from decadal_predictions.utils import *

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

def make_vf_hybC_climatologies(vf_types,vars,Ns,period,smoothing_degrees,clim_years=15,save_clim=True):
    # collect coordinates of the verification to interpolate hindcasts to a matching grid later:
    vf_coords = {}
    for vf_type in vf_types:
        vf_coords[vf_type] = {}
        for var in vars:
            print(f'processing {vf_type} {var}')
            # load verfication dataset (monthly means)
            ds_vf,vf_orig_coords = load_verification(var,period,vf_type=vf_type)
            vf_coords[vf_type][var] = vf_orig_coords
            # coarsen resolution to required number of degrees
            ds_vf_coarse = ds_downsample_to_N_degrees(ds_vf,smoothing_degrees)
            # go through time aggregations:
            for N in Ns:
                # make running average:
                ds_vf_runmean = running_Nyear_mean(ds_vf_coarse,N)
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
                anom_vf_filename = '{0}_hyb{5}_anom_{1}-{2}_{3}yrm_{4}deg.nc'.format(
                    var,period[0],period[-1],N,smoothing_degrees,clim_years
                )
                
                #------------save------------#
                vf_anomalies.to_netcdf(anom_path/anom_vf_filename)
                
                if save_clim:
                    ds_vf_clim.to_netcdf(anom_path/anom_vf_filename.replace('_anom_','_clim_'))


    return vf_coords

def make_hindcast_hybC_anomalies(hc_types,vars,lead_year_ranges,hc_period,smoothing_degrees,ensmem_dim='mem',interp_coords=None,clim_years=15,save_clim=True):

    for hc_type in hc_types:
        for var in vars:
            print(f'processing {hc_type} {var}')
            for lead_year_range in lead_year_ranges:

                ds_hc_lt = get_hindcast_sequence(
                    variable=var,
                    period=hc_period,
                    lead_year_range=lead_year_range,
                    mod_type=hc_type,
                    interpolate_coords=interp_coords[var],
                    N_deg=smoothing_degrees,
                )
                hc_clim = hyb_climatology(ds_hc_lt.mean(ensmem_dim),clim_years=clim_years)

                # compute anomalies:
                hc_anom = ds_hc_lt - hc_clim

                #------------filename------------#
                hc_anom_path = data_paths['processed']/'hindcast/{1}/{0}'.format(
                    var_name_map['long_to_cmip'][var],
                    hc_type
                )
                hc_anom_path.mkdir(parents=True,exist_ok=True)
                hc_anom_filename = '{0}_hyb{1}_anom_{2}-{3}_LY{4}-{5}_{6}deg.nc'.format(
                    var,clim_years,hc_period[0],hc_period[-1],lead_year_range[0],lead_year_range[-1],smoothing_degrees
                )

                #------------save------------#
                hc_anom.to_netcdf(hc_anom_path/hc_anom_filename)
                if save_clim:
                    hc_clim.to_netcdf(hc_anom_path/hc_anom_filename.replace('_anom_','_clim_'))

    return

if __name__ == '__main__':

    vf_types = ['ERA5']
    vf_interp = vf_types[0]
    hc_types = ['NorCPM1']
    vars = ['2m_temperature','total_precipitation','surface_net_solar_radiation','10m_wind_speed']
    Ns = [4,8]
    lead_year_ranges = [[2,9],[2,5],[6,9]]
    period = [1960,2023] # period to load data for
    hc_period = [1960,2018]
    smoothing_degrees = 5
    clim_years = 15

    # compute verification climatologies: should take ~ 25s
    vf_coords = make_vf_hybC_climatologies(vf_types,vars,Ns,period,smoothing_degrees,clim_years)

    # this takes about 18 mins for a single hindcast variable
    # (single combination of (hc_type,var)) on NIRD login node, i.e. for all 4 variables ~ 18 mins * 4 = 72 mins
    make_hindcast_hybC_anomalies(hc_types,vars,lead_year_ranges,hc_period,smoothing_degrees,interp_coords=vf_coords[vf_interp],clim_years=clim_years)

