import xarray as xr
import proplot as pplt
import numpy as np

from decadal_predictions.config import *
from decadal_predictions.utils import lko_std,lko_mean
from decadal_predictions.evaluate import evaluation

spath = data_paths['processed']/'hindcast'

# load NAO index from different models:
lead_year_range = '2-9'
# running year mean:
N = 8
vf_type = 'ERA5'

verification_NAO = xr.open_dataset(data_paths['processed']/f'verification/{vf_type}/indexes/NAO_stat_1960-2018_{N}yrm.nc')

def load_MME(variable,anom=True):

    if variable == 'NAO':
        var_dir = 'indexes'
        anom_ext = '_'
    else:
        var_dir = variable
        if anom:
            anom_ext = 'anom'
        else:
            anom_ext = 'clim'
    all_files = sorted(list(spath.rglob(f'*{var_dir}/{variable}*{anom_ext}*LY{lead_year_range}*nc')))
    all_models_from_config = [fi for fi in all_files if fi.parent.parent.stem in available_models_long]
    models = [fi.parent.parent.stem for fi in all_models_from_config]
    print('\n'.join(models))
    return xr.open_mfdataset(all_models_from_config,combine='nested').sortby('mme_member')


####

### THE BELOW DOES THE RE-SCALING IN A 'FAIR'/LEAVE-K-OUT WAY!
# leaving out inaccessible information of the previous k years
k=4
# select M members with the smallest NAO error wrt to ensemble mean
M = 20

mme = load_MME('NAO').psl.compute()/100
mme_mean = mme.mean('mme_member') # equal weights for each ensemble member!

vf_NAO = verification_NAO.msl/100
# NAO standard deviation in verification:
vf_NAO_lko_std = xr.apply_ufunc(lko_std,vf_NAO,k)
# NAO standard deviations in ensemble mean (predictable signal):
ens_mean_lko_std = xr.apply_ufunc(lko_std,mme_mean,k)
# scaling factor to adjust ens mean NAO variance:
scaling = vf_NAO_lko_std/ens_mean_lko_std # using info that is not available when forecasting!!
# scale ens mean NA):
mme_scaled = mme_mean*scaling
# rolling average to increase ensemble size k times
mme_scaled_smoothed = mme_scaled.rolling(year=k).mean().dropna('year')

# correlation of ERA5 and ensemble mean NAO (lagged ensemble)
print(xr.corr(mme_scaled_smoothed,vf_NAO).values)

# absolute difference between single ensemble members and ensemble mean
# take mean over all years, but do this in cross-validation mode, such that for each year, the mean (and thus the closest members) can be different!
mem_diff = xr.apply_ufunc(lko_mean,abs((mme - mme_scaled_smoothed)),k,input_core_dims=[['year'],[]],output_core_dims=[['year',]],vectorize=True)
mem_axis = mem_diff.get_axis_num('mme_member')
smallest_diff_index = mem_diff.argsort(axis=mem_axis).isel(mme_member=slice(M))
memM_index = []
for y in smallest_diff_index['year']:
    memIX_vals = mem_diff.mme_member.isel(mme_member=smallest_diff_index.sel(year=y).values).values
    memIX_vals_xr = xr.DataArray(
        memIX_vals[:,np.newaxis],
        coords={"rank":np.arange(1,M+1),"year":[y]},
        dims=('rank','year'),
    )
    memM_index.append(memIX_vals_xr)
memM_index = xr.concat(memM_index,dim='year')
mem_filt = mme.sel(year=mem_diff['year'])
mem20 = []
for y in memM_index['year']:
    # mem20.append(mem_filt.sel(year=y).sel(mme_member=memM_index.sel(year=y).values))
    mme_NAO_sel = mem_filt.sel(year=y,mme_member=memM_index.sel(year=y).values).expand_dims({'year':1})
    mme_NAO_sel = mme_NAO_sel.rename({'mme_member':'rank'})
    mme_NAO_sel['rank'] = np.arange(1,M+1)
    mem20.append(mme_NAO_sel)
mem20 = xr.concat(mem20,dim='year').mean('rank')

# mem20 = mme[mem_diff.argsort().isel(mem=slice(20))].mean('mme_member') # this index should then be used to select models in other forecasts!
mem20_lko_std = xr.apply_ufunc(lko_std,mem20,k)
scaling_mem20 = vf_NAO_lko_std/mem20_lko_std # using info that is not available when forecasting!!
mem20_scaled = mem20*scaling_mem20
mem20_scaled_smoothed = mem20_scaled.rolling(year=k).mean()

print(xr.corr(mem20_scaled_smoothed,vf_NAO).values)


fig = pplt.figure(suptitle='NAO LY2-9 prediction',refwidth=5,refheight=3)
ax = fig.subplot(xlabel='x axis', ylabel='y axis')

ax.plot(mme.year,mme,color='grey',alpha=.2,zorder=0,label=None,lw=.5)
ax.plot(mme_mean.year,mme_mean,color='r',lw=.5,ls='dashed',zorder=2)
ax.plot(mme_scaled.year,mme_scaled,color='r',lw=.5,ls='solid',zorder=2)
ax.plot(mme_scaled_smoothed.year,mme_scaled_smoothed,color='r',lw=2,ls='solid',zorder=2)

ax.plot(mem20.year,mem20,color='C0',lw=.5,ls='dashed',zorder=2)
ax.plot(mem20_scaled.year,mem20_scaled,color='C0',lw=.5,ls='solid',zorder=2)
ax.plot(mem20_scaled_smoothed.year,mem20_scaled_smoothed,color='C0',lw=2,ls='solid',zorder=2)

ax.plot(verification_NAO.year,vf_NAO,color='k',lw=2,zorder=3)


### Load the entire fields
for variable,_ in var_name_map['long_to_cf'].items():

    print(variable)

    mme_field = load_MME(variable)
    mme_field = mme_field.reindex(lat=mme_field.lat[::-1])

    vf_name = data_paths['processed']/f'verification/{vf_type}/{variable}/{variable}_hyb15_anom_1960-2023_{N}yrm_5deg.nc'
    ds_vf = xr.open_dataset(vf_name)
    ds_vf = ds_vf.reindex(latitude=ds_vf.latitude[::-1])

    # evaluate 'raw' MME
    scores = evaluation(ds_vf,mme_field,variable,ensmem_dim='mme_member')
    lons = scores['ACC'].lon.values
    lats = scores['ACC'].lat.values

    RMSESS = 1 - scores['RMSE']/scores['STD']

    VAR = ' '.join(variable.split('_')).title()
    levels=pplt.arange(-1,1,.2)

    fig = pplt.figure(refwidth=3)
    axs = fig.subplots(nrows=2, proj='robin')
    # proj = pplt.Proj('robin', lon0=180)
    # axs = pplt.subplots(nrows=2, proj=proj)  # equivalent to above
    axs.format(
        suptitle=f'MME {VAR} (LY{lead_year_range}) scores wrt {vf_type}',
        coast=True, latlines=30, lonlines=60, abc='A.',
        leftlabels=('ACC', 'RMSESS'),
        leftlabelweight='normal',
    )
    pc1 = axs[0].pcolormesh(lons,lats,scores['ACC'].values,levels=levels,cmap='Div')
    fig.colorbar(pc1, loc='r', span=1, label='', extendsize='1.7em')
    pc2 = axs[1].pcolormesh(lons,lats,RMSESS.values,levels=levels,cmap='Div')
    fig.colorbar(pc2, loc='r', span=2, label='', extendsize='1.7em')

    fig.savefig(data_paths['figures_online']/f'scores_{variable}_hyb15_anom_1983-2023_LY{lead_year_range}_5deg_MME_vs_{vf_type}.png')


    ### NAO-match the ensemble:
    mem20_field = []
    for y in memM_index['year']:
        if y.values in mme_field.year.values:
            # print(y.values)
            mme_sel = mme_field.sel(year=y,mme_member=memM_index.sel(year=y).values).expand_dims({'year':1})
            mme_sel = mme_sel.rename({'mme_member':'rank'})
            mme_sel['rank'] = np.arange(1,M+1)
            mem20_field.append(mme_sel)
            # print(smallest_diff_index.sel(year=y).values)
    mem20_field = xr.concat(mem20_field,dim='year')

    # compute rolling average:
    mem20_field = mem20_field.rolling(year=k).mean().dropna('year')

    scores_match = evaluation(ds_vf,mem20_field,variable,ensmem_dim='rank')

    RMSESS_match = 1 - scores_match['RMSE']/scores_match['STD']

    fig = pplt.figure(refwidth=3)
    axs = fig.subplots(nrows=2, proj='robin')
    # proj = pplt.Proj('robin', lon0=180)
    # axs = pplt.subplots(nrows=2, proj=proj)  # equivalent to above
    axs.format(
        suptitle=f'MME {M} NAO matched {VAR} (LY{lead_year_range}) scores wrt {vf_type}',
        coast=True, latlines=30, lonlines=60, abc='A.',
        leftlabels=('ACC', 'RMSESS'),
        leftlabelweight='normal',
    )
    pc1 = axs[0].pcolormesh(lons,lats,scores_match['ACC'].values,levels=levels,cmap='Div')
    fig.colorbar(pc1, loc='r', span=1, label='', extendsize='1.7em')
    pc2 = axs[1].pcolormesh(lons,lats,RMSESS_match.values,levels=levels,cmap='Div')
    fig.colorbar(pc2, loc='r', span=2, label='', extendsize='1.7em')

    fig.savefig(data_paths['figures_online']/f'scores_{variable}_hyb15_anom_1983-2023_LY{lead_year_range}_5deg_MME_NAOmatch{M}_vs_{vf_type}.png')

