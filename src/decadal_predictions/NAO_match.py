# Get the NAO matching relevant information 
# i.e. a ranking of ensemble members in terms of the their NAO MAE

import xarray as xr
import numpy as np

from decadal_predictions.config import *
from decadal_predictions.utils import lko_std,lko_mean,load_MME,ext_from_months

spath = data_paths['processed']/'hindcast'

def nao_match_ensemble(mme,M,seas_ext,lead_year_range):
    NAO_ranking = xr.open_dataarray(data_paths['processed']/f'hindcast/MME_ranking/NAO_match{seas_ext}_ranking_MME_{lead_year_range}_ERA5.nc')
    mem20 = []
    for y in NAO_ranking['year']:
        if y.values in mme.year.values:
            # print(y.values)
            mme_sel = mme.sel(year=y,mme_member=NAO_ranking.sel(year=y).isel(rank=slice(0,M)).values).expand_dims({'year':1})
            mme_sel = mme_sel.rename({'mme_member':'rank'})
            mme_sel['rank'] = np.arange(1,M+1)
            mem20.append(mme_sel)
            # print(smallest_diff_index.sel(year=y).values)
    mem20 = xr.concat(mem20,dim='year')
    return mem20

if __name__ == '__main__':
    # NAO_month_list = [12,1,2,3]
    for NAO_month_list in [[12,1,2,3],[]]:
        NAO_seas_ext = ext_from_months(NAO_month_list)

        # load NAO index from different models:
        for lead_year_range in ['2-5','6-9','2-9']:
            vf_type = 'ERA5'

            ####
            # Verification:
            # running year mean:
            N = int(lead_year_range[-1]) - int(lead_year_range[0])+1
            ref_period = [1960+N,2023] # maximum available period (valid time), can't lie past 2023 (current end of ERA5) and before 1961 (first VALID year in forecast)
            verification_NAO = xr.open_dataset(data_paths['processed']/f'verification/{vf_type}/indexes/NAO_stat{NAO_seas_ext}_1960-2023_{N}yrm.nc')

            vf_NAO = verification_NAO.msl/100
            
            # MME:
            mme = load_MME('NAO',lead_year_range,seas_ext=NAO_seas_ext).psl.compute()/100
            mme_mean = mme.mean('mme_member') # equal weights for each ensemble member!

            # NAO standard deviation in verification:
            vf_NAO_lko_std = xr.apply_ufunc(lko_std,vf_NAO.sel(year=slice(*ref_period)),N)
            vf_NAO_lko_std = vf_NAO_lko_std.interp(year=mme.year, method='nearest', kwargs={"fill_value": None}).ffill('year')

            ### THE BELOW DOES THE RE-SCALING IN A 'FAIR'/LEAVE-K-OUT WAY!
            # NAO standard deviations in ensemble mean (predictable signal):
            ens_mean_lko_std = xr.apply_ufunc(lko_std,mme_mean.sel(year=slice(*ref_period)),N)
            ens_mean_lko_std = ens_mean_lko_std.interp(year=mme.year, method='nearest', kwargs={"fill_value": None}).ffill('year')
            ens_mean_lko_std['iyear'] = mme['iyear']

            # extrapolate  scaling to the full time series:
            vf_NAO_lko_std = vf_NAO_lko_std.interp(year=mme.year, method='nearest', kwargs={"fill_value": None}).ffill('year')
            # scaling factor to adjust ens mean NAO variance:
            scaling = vf_NAO_lko_std/ens_mean_lko_std # using info that is not available when forecasting!!
            # scale ens mean NAO:
            mme_scaled = mme_mean*scaling
            # rolling average to increase ensemble size k times
            mme_scaled_smoothed = mme_scaled.rolling(year=k_lag).mean().dropna('year')

            # save the scaled and lagged MME NAO:
            mme_scaled.to_netcdf(spath/f'MME/indexes/NAO{NAO_seas_ext}_MME_scaled_{vf_type}var_LY{lead_year_range}.nc')
            mme_scaled_smoothed.to_netcdf(spath/f'MME/indexes/NAO{NAO_seas_ext}_MME_scaled_{vf_type}var_lagged{k_lag}_LY{lead_year_range}.nc')

            # correlation of ERA5 and ensemble mean NAO (lagged ensemble)
            print(xr.corr(mme_mean,vf_NAO).values)
            print(xr.corr(mme_scaled_smoothed,vf_NAO).values)

            # absolute difference between single ensemble members and ensemble mean
            mem_diff = abs(mme - mme_scaled)
            mem_axis = mem_diff.get_axis_num('mme_member')
            smallest_diff_index = mem_diff.argsort(axis=mem_axis)
            memM_index = []
            for y in smallest_diff_index['year']:
                memIX_vals = mem_diff.mme_member.isel(mme_member=smallest_diff_index.sel(year=y).values).values
                memIX_vals_xr = xr.DataArray(
                    memIX_vals[:,np.newaxis],
                    coords={"rank":np.arange(1,len(smallest_diff_index.mme_member)+1),"year":[y]},
                    dims=('rank','year'),
                )
                memM_index.append(memIX_vals_xr)
            memM_index = xr.concat(memM_index,dim='year')

            # save ranking:
            memM_index.to_netcdf(spath/f'MME_ranking/NAO_match{NAO_seas_ext}_ranking_MME_{lead_year_range}_{vf_type}.nc')
