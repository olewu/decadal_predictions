# Some generally useful definitions
from pathlib import Path

base_path = Path('/projects/NS9873K')
proj_path = base_path/'owul/projects/pilot4_statkraft'
proc_data_path = base_path/'owul/data/statkraft_pilot4/decadal_predictions'

NAO_box = {
    'south':{
        'lon':slice(332,340),
        'lat':slice(36,40),
        'latitude':slice(40,36),
        },
    'north':{
        'lon':slice(335,344),
        'lat':slice(63,70),
        'latitude':slice(70,63),
        }
} # Azores (28–20° W, 36–40°N) and Iceland (25–16°W, 63–70°N), Smith et al. (2019)


# mapping ERA5 names to DCPP/CMIP6:
var_name_map = {
    "cf_to_cmip":{
        "t2m":"tas",
        "tp":"pr",
        "si10":"sfcWind",
        "ssr":"rsds",
    },
    "long_to_cf":{
        "2m_temperature":"t2m",
        "total_precipitation":"tp",
        "10m_wind_speed":"si10",
        "surface_net_solar_radiation":"ssr",
    },
    "long_to_cmip":{
        "2m_temperature":"tas",
        "total_precipitation":"pr",
        "10m_wind_speed":"sfcWind",
        "surface_net_solar_radiation":"rsds",
    },
    "cmip_to_cf":{
        "tas":"t2m",
        "pr":"tp",
        "sfcWind":"si10",
        "rsds":"ssr",
    },
    "cf_to_long" :{
        't2m':'2m_temperature',
        'tp':'total_precipitation',
        'si10':'10m_wind_speed',
        'ssr':'surface_net_solar_radiation',
    }
}

data_paths = {
    'verification':{
        'ERA5':base_path/'DATA/SFE/ERA5/res_monthly_1',
        # others?
    },
    'hindcast':{
        'NorCPM1':Path('/projects/NS9034K/CMIP6/.cmorout/NorCPM1/dcppA-hindcast'),
        # others?
    },
    'forecast':{
        'NorCPM1':Path('/projects/NS9034K/CMIP6/.cmorout/NorCPM1/dcppB-forecast'),
        # others?
    },
    'processed':proc_data_path,
}