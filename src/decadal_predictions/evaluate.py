import xarray as xr
import proplot as pplt

from decadal_predictions.config import *

def ACC_compute(vf,fc,tdim='year'):
    return xr.corr(fc,vf,dim=tdim)

def RMSE_compute(vf,fc,tdim='year'):
    
    return ((fc - vf)**2).mean(tdim)**.5

def evaluation(vf,fc,var,score_names=['ACC','RMSE','STD'],tdim='year',vf_coord_map={'latitude':'lat','longitude':'lon'}):

    fcem_da = fc[var_name_map['long_to_cmip'][var]].squeeze()
    vf_da = vf.rename(vf_coord_map)[var_name_map['long_to_cf'][var]]

    scores = {}
    if 'ACC' in score_names:
        scores['ACC'] = ACC_compute(vf_da,fcem_da,tdim=tdim)
    
    if 'RMSE' in score_names:
        scores['RMSE'] = RMSE_compute(vf_da,fcem_da,tdim=tdim)
    
    if 'STD' in score_names:
        scores['STD'] = vf_da.std(tdim)

    return scores

if __name__ == '__main__':
    vf_type = 'ERA5'
    # TODO: implement MME
    hc_type = 'CanESM5' # NorCPM1
    variables = ['2m_temperature','total_precipitation']#,'surface_net_solar_radiation','10m_wind_speed']
    Ns = [4,8]
    lead_year_ranges = [[2,9],[2,5],[6,9]]
    vfps,vfpe = vf_period = [1960,2023] # period to load data for
    hcps,hcpe = hc_period = [1960,2018]
    smoothing_degrees = 5
    clim_years = 15

    # var = variables[0]
    # lead_year_range = lead_year_ranges[0]
    for var in variables:
        for lead_year_range in lead_year_ranges:

            lys,lye = lead_year_range
            N = lead_year_range[-1] - lead_year_range[0] + 1

            vf_name = data_paths['processed']/f'verification/{vf_type}/{var}/{var}_hyb{clim_years}_anom_{vfps}-{vfpe}_{N}yrm_{smoothing_degrees}deg.nc'
            hc_name = data_paths['processed']/f'hindcast/{hc_type}/{var}/{var}_hyb{clim_years}_anom_{hcps}-{hcpe}_LY{lys}-{lye}_{smoothing_degrees}deg.nc'

            ds_vf = xr.open_dataset(vf_name)
            ds_vf = ds_vf.reindex(latitude=ds_vf.latitude[::-1])
            ds_hc = xr.open_dataset(hc_name)
            ds_hc = ds_hc.reindex(lat=ds_hc.lat[::-1])

            scores = evaluation(ds_vf,ds_hc,var)
            lons = scores['ACC'].lon.values
            lats = scores['ACC'].lat.values

            RMSESS = 1 - scores['RMSE']/scores['STD']

            VAR = ' '.join(var.split('_')).title()
            levels=pplt.arange(-1,1,.2)

            fig = pplt.figure(refwidth=3)
            axs = fig.subplots(nrows=2, proj='robin')
            # proj = pplt.Proj('robin', lon0=180)
            # axs = pplt.subplots(nrows=2, proj=proj)  # equivalent to above
            axs.format(
                suptitle=f'{hc_type} {VAR} (LY{lys}-{lye}) scores wrt {vf_type}',
                coast=True, latlines=30, lonlines=60, abc='A.',
                leftlabels=('ACC', 'RMSESS'),
                leftlabelweight='normal',
            )
            pc1 = axs[0].pcolormesh(lons,lats,scores['ACC'].values,levels=levels,cmap='Div')
            fig.colorbar(pc1, loc='r', span=1, label='', extendsize='1.7em')
            pc2 = axs[1].pcolormesh(lons,lats,RMSESS.values,levels=levels,cmap='Div')
            fig.colorbar(pc2, loc='r', span=2, label='', extendsize='1.7em')

            fig.savefig(data_paths['figures_online']/f'hindcast_scoers/scores_{var}_hyb{clim_years}_anom_{hcps}-{hcpe}_LY{lys}-{lye}_{smoothing_degrees}deg_{hc_type}_vs_{vf_type}.png')
