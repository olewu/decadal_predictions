import xarray as xr
import proplot as pplt
from matplotlib.colors import BoundaryNorm
import numpy as np

from decadal_predictions.config import *
from decadal_predictions.utils import lko_std,load_MME,ext_from_months
from decadal_predictions.evaluate import evaluation
from decadal_predictions.NAO_match import nao_match_ensemble

import math

def nice_axis_limits(data, num_steps=11, padding=0.1):
    """
    Determines a 'nice' axis range with a fixed number of steps and symmetric ranges around zero if possible.

    Args:
        data (list of float): The data points to be displayed.
        num_steps (int): Desired number of steps on the axis.
        padding (float): Fraction of range to extend the limits (default 10%).

    Returns:
        tuple: (nice_min, nice_max, step_size) rounded axis limits and step size.
    """

    data_min, data_max = min(data), max(data)

    # Compute raw range
    raw_range = data_max - data_min

    if raw_range == 0:  # Handle single-value case
        step = 10 ** math.floor(math.log10(abs(data_min) if data_min != 0 else 1))
        return data_min - step, data_max + step, step

    # Apply padding
    padded_range = raw_range * (1 + 2 * padding)

    # Determine step size
    step_size = 10 ** math.floor(math.log10(padded_range / num_steps))

    # Adjust step size to be a round number (1, 2, or 5 times a power of 10)
    for factor in [1, 2, 5, 10]:  # Prefer 1, 2, 5 as step multipliers
        # print(factor)
        if padded_range / (step_size * factor) >= num_steps:
            step_size *= factor
            break

    # Decide if we should make the range symmetric around 0
    if data_min < 0 and data_max > 0:
        # If both negative and positive values exist, check if zero can be in the center
        if abs(data_min) < padded_range / 2 and abs(data_max) < padded_range / 2:
            # Make the range symmetric around zero
            nice_max = max(abs(data_min), abs(data_max))
            nice_max = math.ceil(nice_max / step_size) * step_size
            nice_min = -nice_max
        else:
            nice_min = math.floor(data_min / step_size) * step_size
            nice_max = math.ceil(data_max / step_size) * step_size
    else:
        # Handle case where range is all positive or negative
        nice_min = math.floor(data_min / step_size) * step_size
        nice_max = math.ceil(data_max / step_size) * step_size

    return nice_min, nice_max, step_size

nsteps = 11

spath = data_paths['processed']/'hindcast'

# load NAO index from different models:
lead_year_ranges = [[2,5],[6,9],[2,9]]
vf_type = 'ERA5'
# select M members with the smallest NAO error wrt to ensemble mean
M = 20
prnt=False

# Eur/Atl domain:
lat_sel = slice(55,70)
lon_sel = slice(-10,25)

for NAO_month_list in [[12,1,2,3],[]]:
# NAO_month_list = [12,1,2,3]
    NAO_seas_ext = ext_from_months(NAO_month_list)

    for lyr in lead_year_ranges:
        # lyr = lead_year_ranges[-1]
        lead_year_range = '{}-{}'.format(lyr[0],lyr[-1])
        print(lead_year_range)

        # running year mean:
        N = lyr[-1] - lyr[0] + 1
        ref_period = [1960+N,2023]
    
        verification_NAO = xr.open_dataset(data_paths['processed']/f'verification/{vf_type}/indexes/NAO_stat{NAO_seas_ext}_1961-2023_{N}yrm.nc')
        vf_NAO = verification_NAO.msl/100

        # Load the MME (already scaled and smoothed)
        mme_scaled = xr.open_dataarray(spath/f'MME/indexes/NAO{NAO_seas_ext}_MME_scaled_{vf_type}var_LY{lead_year_range}.nc')
        mme_scaled_smoothed = xr.open_dataarray(spath/f'MME/indexes/NAO{NAO_seas_ext}_MME_scaled_{vf_type}var_lagged{k_lag}_LY{lead_year_range}.nc')

        # Load raw MME:
        mme = load_MME('NAO',lead_year_range,seas_ext=NAO_seas_ext).psl.compute()/100
        mme_mean = mme.mean('mme_member') # equal weights for each ensemble member!

        # correlation of ERA5 and ensemble mean NAO (lagged ensemble)
        raw_mme_corr = xr.corr(mme_mean,vf_NAO).values
        lagged_mme_corr = xr.corr(mme_scaled_smoothed,vf_NAO).values

        if prnt:
            print(raw_mme_corr)
            print(lagged_mme_corr)

        # get the NAO-matched ensemble:
        mem20 = nao_match_ensemble(mme,M,NAO_seas_ext,lead_year_range)
        mem20_mean = mem20.mean('rank')

        # NAO standard deviation in verification for scaling factor:
        vf_NAO_lko_std = xr.apply_ufunc(lko_std,vf_NAO.sel(year=slice(*ref_period)),N)
        vf_NAO_lko_std = vf_NAO_lko_std.interp(year=mme.year, method='nearest', kwargs={"fill_value": None}).ffill('year')

        # # NAO standard deviations in ensemble mean (predictable signal):
        # ens_mean_lko_std = xr.apply_ufunc(lko_std,mme_mean.sel(year=slice(*ref_period)),N)
        # ens_mean_lko_std = ens_mean_lko_std.interp(year=mme.year, method='nearest', kwargs={"fill_value": None}).ffill('year')
        # ens_mean_lko_std['iyear'] = mme['iyear']

        # # scaling factor to adjust ens mean NAO variance:
        # scaling = vf_NAO_lko_std/ens_mean_lko_std # not using info that is not available when forecasting!!
        # # scale ens mean NAO:
        # mme_scaled = mme_mean*scaling
        # # rolling average to increase ensemble size k times
        # mme_scaled_smoothed = mme_scaled.rolling(year=k_lag).mean().dropna('year')


        # absolute difference between single ensemble members and ensemble mean
        # take mean over all years, but do this in cross-validation mode, such that for each year, the mean (and thus the closest members) can be different!
        # mem_diff = xr.apply_ufunc(lko_mean,abs((mme - mme_mean)),k,input_core_dims=[['year'],[]],output_core_dims=[['year',]],vectorize=True)
        # mem_diff = abs(mme - mme_mean)
        # mem_axis = mem_diff.get_axis_num('mme_member')
        # smallest_diff_index = mem_diff.argsort(axis=mem_axis).isel(mme_member=slice(M))
        # memM_index = []
        # for y in smallest_diff_index['year']:
        #     memIX_vals = mem_diff.mme_member.isel(mme_member=smallest_diff_index.sel(year=y).values).values
        #     memIX_vals_xr = xr.DataArray(
        #         memIX_vals[:,np.newaxis],
        #         coords={"rank":np.arange(1,M+1),"year":[y]},
        #         dims=('rank','year'),
        #     )
        #     memM_index.append(memIX_vals_xr)
        # memM_index = xr.concat(memM_index,dim='year')
        # mem_filt = mme.sel(year=mem_diff['year'])
        # mem20 = []
        # for y in memM_index['year']:
        #     # mem20.append(mem_filt.sel(year=y).sel(mme_member=memM_index.sel(year=y).values))
        #     mme_NAO_sel = mem_filt.sel(year=y,mme_member=memM_index.sel(year=y).values).expand_dims({'year':1})
        #     mme_NAO_sel = mme_NAO_sel.rename({'mme_member':'rank'})
        #     mme_NAO_sel['rank'] = np.arange(1,M+1)
        #     mem20.append(mme_NAO_sel)
        # mem20 = xr.concat(mem20,dim='year').mean('rank')

        # mem20 = mme[mem_diff.argsort().isel(mem=slice(20))].mean('mme_member') # this index should then be used to select models in other forecasts!
        mem20_lko_std = xr.apply_ufunc(lko_std,mem20_mean,N)

        scaling_mem20 = vf_NAO_lko_std/mem20_lko_std
        mem20_scaled = mem20_mean*scaling_mem20
        mem20_scaled_smoothed = mem20_scaled.rolling(year=k_lag).mean().dropna('year')

        # correlation of ERA5 and ensemble mean NAO (lagged ensemble)
        raw_mme20_corr = xr.corr(mem20,vf_NAO).values
        lagged_mme20_corr = xr.corr(mem20_scaled_smoothed,vf_NAO).values

        if prnt:
            print(raw_mme20_corr)
            print(lagged_mme20_corr)


        qint = [.05,.95]
        mme_90pint = mme.quantile(qint,dim='mme_member')
        # mme_scaled_smoothed_90pint = mem20_scaled.rolling(year=k).construct('window_dim').quantile(qint,dim=['rank','window_dim'])

        fig = pplt.figure(refheight=2,refwidth=4)
        axs = fig.subplots(ncols=1)
        axs.format(
            suptitle=f'NAO prediction LY{lead_year_range}',
            leftlabelweight='normal',#xlim=(1968,2026),ylim=(-11.5,9.5),
        )
        axs[0].set_ylabel('sea level pressure anomaly [hPa]')
        
        # fig.savefig('/projects/NS9873K/owul/figures/decadal_predictions/NAO_pred/NAO_1.png')
        # ax.plot(mme.year,mme,color='grey',alpha=.2,zorder=0,label=None,lw=.5)
        axs[0].plot(verification_NAO.year,vf_NAO,color='k',lw=2,zorder=3,label='ERA5')
        # fig.savefig('/projects/NS9873K/owul/figures/decadal_predictions/NAO_pred/NAO_2.png')
        axs[0].fill_between(mme_90pint.year,mme_90pint.sel(quantile=qint[0]),mme_90pint.sel(quantile=qint[1]),color='r',alpha=.2,zorder=0,label='_')
        axs[0].plot(mme_mean.year,mme_mean,color='r',lw=.5,ls='dashed',zorder=2,label=f'MME raw: {raw_mme_corr:.2f}')
        # fig.savefig('/projects/NS9873K/owul/figures/decadal_predictions/NAO_pred/NAO_3.png')

        # axs[1].fill_between(mme_scaled_smoothed_90pint.year,mme_scaled_smoothed_90pint.sel(quantile=qint[0]),mme_scaled_smoothed_90pint.sel(quantile=qint[1]),color='r',alpha=.2,zorder=0,label=None)
        axs[0].plot(mme_scaled.year,mme_scaled,color='r',lw=.5,ls='solid',zorder=2)
        # fig.savefig('/projects/NS9873K/owul/figures/decadal_predictions/NAO_pred/NAO_4.png')
        axs[0].plot(mme_scaled_smoothed.year,mme_scaled_smoothed,color='r',lw=2,ls='solid',zorder=2,label=f'MME lagged: {lagged_mme_corr:.2f}')
        # fig.savefig('/projects/NS9873K/owul/figures/decadal_predictions/NAO_pred/NAO_5.png')

        axs[0].legend(loc='lower right',ncol=1,fontsize=9)
        # add a box with the correlation values in the plot. the box has the title ACC and the lists the correlation values in two seperate lines:
        # axs[0].text(0.975,0.05,f'ACC\nraw: {raw_mme_corr:.2f}\nlagged: {lagged_mme_corr:.2f}', # \nmatched: {raw_mme20_corr:.2f}\nlagged matched: {lagged_mme20_corr:.2f}
        #     transform=axs[0].transAxes,fontsize=9,va='bottom',ha='right',bbox=dict(facecolor='white', alpha=0.5))
        # fig.savefig('/projects/NS9873K/owul/figures/decadal_predictions/NAO_pred/NAO_6.png')

        axs[0].plot(mem20_mean.year,mem20_mean,color='C0',lw=.5,ls='dashed',zorder=2)
        axs[0].plot(mem20_scaled.year,mem20_scaled,color='C0',lw=.5,ls='solid',zorder=2)
        axs[0].plot(mem20_scaled_smoothed.year,mem20_scaled_smoothed,color='C0',lw=2,ls='solid',zorder=2)


        fig.savefig(data_paths['figures_online']/f'NAO/NAO_stat{NAO_seas_ext}_init_1960-2018_LY{lead_year_range}_MME_vs_{vf_type}.png')

        # could optionally loop over different field averaging seasons (e.g. sort annual mean field by winter NAO):
        target_months = NAO_month_list
        seas_ext = ext_from_months(target_months)

        for variable in var_name_map['long_to_cf'].keys():
            for clim_type in ['standard_l1o','hyb15']:
                print(variable)

                UNIT = units[var_name_map['long_to_cf'][variable]]

                # mme_field = xr.open_dataset(data_paths['processed']/f'hindcast/MME/{variable}/{variable}_{clim_type}_anom_1960-2018_LY{lead_year_range}_5deg_all{seas}.nc')
                mme_field = load_MME(variable,lead_year_range,seas_ext=seas_ext,clim_type=clim_type).compute() * unit_conversion['DCPP'][var_name_map['long_to_cf'][variable]]
                mme_field = mme_field.reindex(lat=mme_field.lat[::-1])
                # TODO: standardization should be leave-k-out:
                mme_mean_stdz = mme_field.mean('mme_member') # /mme_field.mean('mme_member').std('year')
                mme_field_smoothed = mme_mean_stdz.rolling(year=k_lag).mean()

                vf_name = data_paths['processed']/f'verification/{vf_type}/{variable}/{variable}_{clim_type}_anom_1960-2023_{N}yrm_5deg{seas_ext}.nc'
                ds_vf = xr.open_dataset(vf_name) * unit_conversion['ERA5'][var_name_map['long_to_cf'][variable]]
                if variable == 'surface_net_solar_radiation':
                    ds_vf = ds_vf.sel(year=slice(1960,2022))
                # standardize verification field (TODO: should be leave-k-out):
                ds_vf_stdz = ds_vf # /ds_vf.std('year')

                # evaluate 'raw' MME
                scores = evaluation(ds_vf_stdz,mme_field_smoothed,variable)
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
                    suptitle=f'{VAR} MME (LY{lead_year_range})',
                    coast=True, latlines=30, lonlines=60, abc='A.',
                    leftlabels=('ACC', 'RMSESS'),
                    leftlabelweight='normal',
                )
                pc1 = axs[0].pcolormesh(lons,lats,scores['ACC'].values,levels=levels,cmap='Div',cmap_kw={'cut': -.2})
                fig.colorbar(pc1, loc='r', span=1, label='', extendsize='1.7em')
                pc2 = axs[1].pcolormesh(lons,lats,RMSESS.values,levels=levels,cmap='Div',cmap_kw={'cut': -.2})
                fig.colorbar(pc2, loc='r', span=2, label='', extendsize='1.7em')

                # plot the outline of Eur/Atl domain:
                axs[0].plot([lon_sel.start,lon_sel.stop,lon_sel.stop,lon_sel.start,lon_sel.start],[lat_sel.start,lat_sel.start,lat_sel.stop,lat_sel.stop,lat_sel.start],color='lightgreen',linewidth=1)
                axs[1].plot([lon_sel.start,lon_sel.stop,lon_sel.stop,lon_sel.start,lon_sel.start],[lat_sel.start,lat_sel.start,lat_sel.stop,lat_sel.stop,lat_sel.start],color='lightgreen',linewidth=1)

                fig.savefig(data_paths['figures_online']/f'MME_scores/{variable}_{clim_type}_anom_init_1960-2018{seas_ext}_LY{lead_year_range}_5deg_MME_vs_{vf_type}.png')


                ### load NAO-matched ensemble:
                mem20_field = nao_match_ensemble(mme_field,M,NAO_seas_ext,lead_year_range)
                # TODO: standardization should be leave-k-out:
                mem20_field_stdz = mem20_field.mean('rank') # /mem20_field.mean('rank').std('year')
                mem20_field_smoothed = mem20_field_stdz.rolling(year=k_lag).mean()

                scores_match = evaluation(ds_vf_stdz,mem20_field_smoothed,variable)

                RMSESS_match = 1 - scores_match['RMSE']/scores_match['STD']

                fig = pplt.figure(refwidth=3)
                axs = fig.subplots(nrows=2, proj='robin')
                axs.format(
                    suptitle=f'{VAR} NAO-matched (LY{lead_year_range})',
                    coast=True, latlines=30, lonlines=60, abc='A.',
                    leftlabels=('ACC', 'RMSESS'),
                    leftlabelweight='normal',
                )
                pc1 = axs[0].pcolormesh(lons,lats,scores_match['ACC'].values,levels=levels,cmap='Div',cmap_kw={'cut': -.2})
                fig.colorbar(pc1, loc='r', span=1, label='', extendsize='1.7em')
                pc2 = axs[1].pcolormesh(lons,lats,RMSESS_match.values,levels=levels,cmap='Div',cmap_kw={'cut': -.2})
                fig.colorbar(pc2, loc='r', span=2, label='', extendsize='1.7em')

                axs[0].plot([lon_sel.start,lon_sel.stop,lon_sel.stop,lon_sel.start,lon_sel.start],[lat_sel.start,lat_sel.start,lat_sel.stop,lat_sel.stop,lat_sel.start],color='lightgreen',linewidth=1)
                axs[1].plot([lon_sel.start,lon_sel.stop,lon_sel.stop,lon_sel.start,lon_sel.start],[lat_sel.start,lat_sel.start,lat_sel.stop,lat_sel.stop,lat_sel.start],color='lightgreen',linewidth=1)

                fig.savefig(data_paths['figures_online']/f'MME_scores/{variable}_NAO{NAO_seas_ext}_matched_{M}_{clim_type}_anom_init_1960-2018{seas_ext}_LY{lead_year_range}_5deg_MME_vs_{vf_type}.png')

                # Plot difference in skill between MME and NAO-matched ensemble:

                fig = pplt.figure(refwidth=3)
                axs = fig.subplots(nrows=2, proj='robin')
                axs.format(
                    suptitle=f'{VAR} (LY{lead_year_range})' + ' skill difference',
                    coast=True, latlines=30, lonlines=60, abc='A.',
                    leftlabels=('ACC', 'RMSESS'),
                    leftlabelweight='normal',
                )
                pc1 = axs[0].pcolormesh(lons,lats,scores_match['ACC'].values - scores['ACC'].values,levels=pplt.arange(-.4,.4,.1),cmap='Div',extend='both')
                fig.colorbar(pc1, loc='r', span=1, label='', extendsize='1.7em')
                pc2 = axs[1].pcolormesh(lons,lats,RMSESS_match.values - RMSESS.values,levels=pplt.arange(-.4,.4,.1),cmap='Div',extend='both')
                fig.colorbar(pc2, loc='r', span=2, label='', extendsize='1.7em')

                axs[0].plot([lon_sel.start,lon_sel.stop,lon_sel.stop,lon_sel.start,lon_sel.start],[lat_sel.start,lat_sel.start,lat_sel.stop,lat_sel.stop,lat_sel.start],color='lightgreen',linewidth=1)
                axs[1].plot([lon_sel.start,lon_sel.stop,lon_sel.stop,lon_sel.start,lon_sel.start],[lat_sel.start,lat_sel.start,lat_sel.stop,lat_sel.stop,lat_sel.start],color='lightgreen',linewidth=1)

                fig.savefig(data_paths['figures_online']/f'MME_scores/{variable}_skill_diff_NAO{NAO_seas_ext}_matched_{M}_{clim_type}_anom_init_1960-2018{seas_ext}_LY{lead_year_range}_5deg_MME_vs_{vf_type}.png')

                # Plot two example years, one with high NAO and one with low NAO:
                # high NAO year:
                HIGH_YEAR = int(vf_NAO.sel(year=slice(1980,2023)).idxmax().values)
                LOW_YEAR = int(vf_NAO.sel(year=slice(1980,2023)).idxmin().values)

                # Plot an example hindcast of the field:
                for hilo,EX_YEAR in zip(['high','low'],[HIGH_YEAR,LOW_YEAR]):
                    fig = pplt.figure(refwidth=3)
                    axs = fig.subplots(nrows=3, proj='robin')
                    axs.format(
                        suptitle=f'{VAR} (LY{lead_year_range}) {EX_YEAR}',
                        coast=True, latlines=30, lonlines=60, abc='A.',
                        leftlabels=(f'{vf_type}','MME', f'MME {M} NAO-matched'),
                        leftlabelweight='normal',
                    )
                    bmin,bmax,stepsize = nice_axis_limits(ds_vf_stdz.sel(year=EX_YEAR)[var_name_map['long_to_cf'][variable]].values.flatten(),num_steps=nsteps)
                    bnorm = BoundaryNorm(np.linspace(bmin,bmax,nsteps),ncolors=256, extend='both')
                    pc0 = axs[0].pcolormesh(lons,lats,ds_vf_stdz.sel(year=EX_YEAR)[var_name_map['long_to_cf'][variable]],norm=bnorm)
                    fig.colorbar(pc0, loc='r', span=1, label=f'[{UNIT}]', extendsize='1.7em')
                    pc1 = axs[1].pcolormesh(lons,lats,mme_field_smoothed.sel(year=EX_YEAR)[var_name_map['long_to_cmip'][variable]],norm=bnorm)
                    fig.colorbar(pc1, loc='r', span=2, label=f'[{UNIT}]', extendsize='1.7em')
                    pc2 = axs[2].pcolormesh(lons,lats,mem20_field_smoothed.sel(year=EX_YEAR)[var_name_map['long_to_cmip'][variable]],norm=bnorm)
                    fig.colorbar(pc2, loc='r', span=3, label=f'[{UNIT}]', extendsize='1.7em')

                    fig.savefig(data_paths['figures_online']/f'MME_example_hc/{variable}_hindcast_{M}_{hilo}NAO{NAO_seas_ext}_{clim_type}_anom_init_1960-2018{seas_ext}_LY{lead_year_range}_5deg_MME_vs_{vf_type}.png')


                # lat_sel = 57.5
                # lon_sel = -2.5

                # mme_sel = mme_mean_stdz.sel(lat=lat_sel,lon=lon_sel)[var_name_map['long_to_cmip'][variable]]
                # mme_smoothed_sel = mme_field_smoothed.sel(lat=lat_sel,lon=lon_sel)[var_name_map['long_to_cmip'][variable]]
                # mme20_sel = mem20_field_stdz.sel(lat=lat_sel,lon=lon_sel)[var_name_map['long_to_cmip'][variable]]
                # mem20_smoothed_sel = mem20_field_smoothed.sel(lat=lat_sel,lon=lon_sel)[var_name_map['long_to_cmip'][variable]]
                # vf_sel = ds_vf_stdz.sel(latitude=lat_sel,longitude=lon_sel)[var_name_map['long_to_cf'][variable]]

                # fig = pplt.figure(suptitle=f'{VAR} LY{lead_year_range} prediction',refwidth=5,refheight=3)
                # ax = fig.subplot(xlabel='year', ylabel=f'{VAR} standardized [1]')

                # ax.plot(mme_sel.year,mme_sel,color='r',lw=.5,ls='dashed',zorder=2)
                # # ax.plot(mme_scaled.year,mme_scaled,color='r',lw=.5,ls='solid',zorder=2)
                # ax.plot(mme_smoothed_sel.year,mme_smoothed_sel,color='r',lw=2,ls='solid',zorder=2)

                # ax.plot(mme20_sel.year,mme20_sel,color='C0',lw=.5,ls='dashed',zorder=2)
                # # ax.plot(mem20_scaled.year,mem20_scaled,color='C0',lw=.5,ls='solid',zorder=2)
                # ax.plot(mem20_smoothed_sel.year,mem20_smoothed_sel,color='C0',lw=2,ls='solid',zorder=2)


                # ax.plot(vf_sel.year,vf_sel,color='k',lw=2,zorder=3)

                # print(scores['ACC'].sel(lat=lat_sel,lon=lon_sel).values)
                # print(scores_match['ACC'].sel(lat=lat_sel,lon=lon_sel).values)


                # Make average over European domain:

                # take the mme (unprocessed) and build domain average:
                mme_domain = mme_field.sel(lat=lat_sel,lon=lon_sel).mean(['lat','lon'])[var_name_map['long_to_cmip'][variable]]
                mme_domain_mean_stdz = mme_domain.mean('mme_member') # /mme_domain.mean('mme_member').std('year')
                mme_domain_smoothed_sel = mme_domain_mean_stdz.rolling(year=k_lag).mean()

                mme20_domain = mem20_field.sel(lat=lat_sel,lon=lon_sel).mean(['lat','lon'])[var_name_map['long_to_cmip'][variable]]
                mme20_domain_mean_stdz = mme20_domain.mean('rank') # /mme20_domain.mean('rank').std('year')
                mem20_domain_smoothed_sel = mme20_domain_mean_stdz.rolling(year=k_lag).mean()

                vf_sel = ds_vf_stdz.sel(latitude=lat_sel,longitude=lon_sel).mean(['latitude','longitude'])[var_name_map['long_to_cf'][variable]]

                var_corr_atl_mme = xr.corr(mme_domain_smoothed_sel,vf_sel).values
                var_corr_atl_mme_matched = xr.corr(mem20_domain_smoothed_sel,vf_sel).values
                
                if prnt:
                    print(var_corr_atl_mme)
                    print(var_corr_atl_mme_matched)

                # Plot all domain averaged time series into a single figure:
                fig = pplt.figure(suptitle=f'{VAR} LY{lead_year_range} prediction',refwidth=4,refheight=2)
                ax = fig.subplot(xlabel='year', ylabel=f'{VAR} [{UNIT}]')
                ax.plot(mme_domain_mean_stdz.year,mme_domain_mean_stdz,color='r',lw=.5  ,ls='dashed',zorder=2)
                ax.plot(mme_domain_smoothed_sel.year,mme_domain_smoothed_sel,color='r',lw=2,ls='solid',zorder=2)
                ax.plot(mme20_domain_mean_stdz.year,mme20_domain_mean_stdz,color='C0',lw=.5,ls='dashed',zorder=2)
                ax.plot(mem20_domain_smoothed_sel.year,mem20_domain_smoothed_sel,color='C0',lw=2,ls='solid',zorder=2)
                ax.plot(vf_sel.year,vf_sel,color='k',lw=2,zorder=3)

                # print correlation values inside a box in the plot:
                ax.text(0.975,0.05,f'ACC\nlagged: {var_corr_atl_mme:.2f}\nNAO-matched: {var_corr_atl_mme_matched:.2f}',
                    transform=ax.transAxes,fontsize=9,va='bottom',ha='right',bbox=dict(facecolor='white', alpha=0.5))
                # save figure:
                fig.savefig(data_paths['figures_online']/f'MME_example_hc/{variable}_atl_domain_NAO{NAO_seas_ext}_{M}_{clim_type}_anom_init_1960-2018{seas_ext}_LY{lead_year_range}_5deg_MME_vs_{vf_type}.png')
