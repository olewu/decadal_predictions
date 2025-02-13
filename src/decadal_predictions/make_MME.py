# Create a 'raw' MME for each variable

from decadal_predictions.utils import get_hindcast_sequence, load_verification, ds_downsample_to_N_degrees, adjust_latlon, running_Nyear_mean, make_season_group
from decadal_predictions.config import *

import xarray as xr

lead_year_ranges = [[2,5],[6,9],[2,9]]

ref_period = [1960,2018]
N_deg=5
season = [12,1,2,3]
season_ext = '_DJFM'

for variable in list(var_name_map['long_to_cf'].keys())[1:2]:
    for lead_year_range in lead_year_ranges[-1:]:
        
        print(f'processing {variable} {lead_year_range}')

        ds = []
        for hc_type in available_models_long:
            ds.append(
                get_hindcast_sequence(
                    variable,
                    ref_period,
                    lead_year_range=lead_year_range,
                    mod_type=hc_type,
                    N_deg=N_deg,
                    months_in_season=season,
                    mem_dim='mme_member'
                )
            )

        MME = xr.concat(ds,dim='mme_member').sortby('mme_member')

        MME_path = data_paths['processed']/'hindcast/MME/{0}'.format(
            variable
        )

        MME_path.mkdir(parents=True,exist_ok=True)
        MME_filename = '{0}_raw_{1}-{2}_LY{3}-{4}_{5}deg{6}.nc'.format(
            variable,ref_period[0],ref_period[-1],lead_year_range[0],lead_year_range[-1],N_deg,season_ext
        )

        #------------save------------#
        MME.to_netcdf(MME_path/MME_filename)


#------------make the corresponding verification dataset------------#
Ns = [4,8]
vf_period = [1960,2023]
vf_type = 'ERA5'
for variable in var_name_map['long_to_cf'].keys():
    # if variable == '2m_temperature':
    #     continue
    ds_vf = load_verification(variable,vf_period,vf_type=vf_type)
    ds_vf = adjust_latlon(ds_vf)
    # vf_coords[vf_type][variable] = ds_vf.coords
    # coarsen resolution to required number of degrees
    ds_vf_coarse = ds_downsample_to_N_degrees(ds_vf,N_deg)
    if season is not None:
        seas_grp = make_season_group(ds_vf_coarse,months_in_season=season)
        ds_vf_coarse = ds_vf_coarse.groupby(seas_grp).mean('time').sel(year=slice(vf_period[0]-1,vf_period[-1]+1))
        t_coord = 'year'
    else:
        t_coord = 'time'
    # go through time aggregations:
    for N in Ns:
        # make running average:
        ds_vf_runmean = running_Nyear_mean(ds_vf_coarse,N,time_coord=t_coord)
        # compute the hybrid-15 climatology
        
        #------------filename------------#
        vf_path = data_paths['processed']/'verification/{0}/{1}'.format(
            vf_type,variable
        )
        # make path if it doesn't exist:
        vf_path.mkdir(parents=True,exist_ok=True)
        # build filename:
        vf_filename = '{0}_raw_{1}-{2}_{3}yrm_{4}deg{5}.nc'.format(
            variable,vf_period[0],vf_period[-1],N,N_deg,season_ext
        )
        
        #------------save------------#
        ds_vf_runmean.to_netcdf(vf_path/vf_filename)
        