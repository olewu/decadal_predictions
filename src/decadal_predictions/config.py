# Some generally useful definitions
from pathlib import Path
import os

# proj_base = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]
base_path = Path('/projects/NS9873K')

# lagged ensemble factor 
k_lag = 4

example_grid_file = '/projects/NS9873K/DATA/SFE/ERA5/res_6hrly_1/2m_temperature/2m_temperature_2000_01.nc'

NAO_box = {
    'south':{
        'lon':slice(-28,-20),
        'longitude':slice(332,340),
        'lat':slice(36,40),
        'latitude':slice(40,36),
        },
    'north':{
        'lon':slice(-25,-16),
        'longitude':slice(335,344),
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
        "msl":"psl",
    },
    "long_to_cf":{
        "2m_temperature":"t2m",
        "total_precipitation":"tp",
        "10m_wind_speed":"si10",
        "surface_net_solar_radiation":"ssr",
        "mean_sea_level_pressure":"msl",
    },
    "long_to_cmip":{
        "2m_temperature":"tas",
        "total_precipitation":"pr",
        "10m_wind_speed":"sfcWind",
        "surface_net_solar_radiation":"rsds",
        "mean_sea_level_pressure":"psl",
    },
    "cmip_to_cf":{
        "tas":"t2m",
        "pr":"tp",
        "sfcWind":"si10",
        "rsds":"ssr",
        "psl":"msl",
    },
    "cf_to_long" :{
        "t2m":"2m_temperature",
        "tp":"total_precipitation",
        "si10":"10m_wind_speed",
        "ssr":"surface_net_solar_radiation",
        "msl":"mean_sea_level_pressure",
    },
    "cds_to_cmip":{
        "sea_level_pressure":"psl",
        "precipitation":"pr",
        "near_surface_air_temperature":"tas",
    }
}

unit_conversion = {
    'ERA5':{
        "t2m":1,
        "tp":1000, # to get from m to mm
        "si10":1,
        "ssr":1/86400, # to get from J/s to W/m^2
        "msl":1/100, # to get from Pa to hPa
    },
    'DCPP':{
        "t2m":1,
        "tp":86400, # to get from kg/m2/s to mm
        "si10":1,
        "ssr":1,
        "msl":1/100, # to get from Pa to hPa
    }
}

units = {
    "t2m":"˚C",
    "tp":"mm",
    "si10":"m/s",
    "ssr":"W/m^2",
    "msl":"hPa",
}

data_paths = {
    'verification':{
        'ERA5':base_path/'DATA/SFE/ERA5/res_monthly_1',
        # others?
    },
    'hindcast':{
        'NorCPM1':Path('/datalake/NS9873K/DATA/DCPP/dcppA-hindcast/NorCPM1/'),
        'EC-Earth3':Path('/datalake/NS9873K/DATA/DCPP/dcppA-hindcast/EC-Earth3/'),
        'HadGEM3-GC31-MM':Path('/datalake/NS9873K/DATA/DCPP/dcppA-hindcast/HadGEM3-GC31-MM/'),
        'CMCC-CM2-SR5':Path('/datalake/NS9873K/DATA/DCPP/dcppA-hindcast/CMCC-CM2-SR5/'),
        'MPI-ESM1-2-HR':Path('/datalake/NS9873K/DATA/DCPP/dcppA-hindcast/MPI-ESM1-2-HR/'),
        'CanESM5':Path('/datalake/NS9873K/DATA/DCPP/dcppA-hindcast/CanESM5/'),
        'MPI-ESM1-2-LR':Path('/datalake/NS9873K/DATA/DCPP/dcppA-hindcast/MPI-ESM1-2-LR/'),
        'CNRM-ESM2-1':Path('/datalake/NS9873K/DATA/DCPP/dcppA-hindcast/CNRM-ESM2-1/'),
        'MIROC6':Path('/datalake/NS9873K/DATA/DCPP/dcppA-hindcast/MIROC6/'),
    },
    'forecast':{
        'NorCPM1':Path('/projects/NS9034K/CMIP6/.cmorout/NorCPM1/dcppB-forecast'),
        # others?
    },
    'processed':base_path/'owul/data/statkraft_pilot4/decadal_predictions',
    'figures':base_path/'owul/figures/decadal_predictions',
    'figures_online':base_path/'www/decadal'
}

available_models = [
    "norcpm1",
    "cmcc_cm2_sr5",
    "ec_earth3",
    "hadgem3_gc31_mm",
    "mpi_esm1_2_hr",
    "canesm5",
    # "mpi_esm1_2_lr",
    # "cnrm_esm2_1",
    # "miroc6",
]

model_name_map = {
    "norcpm1":"NorCPM1",
    "ec_earth3":"EC-Earth3",
    "hadgem3_gc31_mm":"HadGEM3-GC31-MM",
    "cmcc_cm2_sr5":"CMCC-CM2-SR5",
    "mpi_esm1_2_hr":"MPI-ESM1-2-HR",
    "canesm5":"CanESM5",
    # "mpi_esm1_2_lr":"MPI-ESM1-2-LR",
    # "cnrm_esm2_1":"CNRM-ESM2-1",
    # "miroc6":"MIROC6",
}

available_models_long = [model_name_map[mod] for mod in available_models]

model_name_map_r = {val:key for key,val in model_name_map.items()}

expected_hindcast_ens_size = {
    "NorCPM1":20,
    "EC-Earth3":16,
    "HadGEM3-GC31-MM":10,
    "CMCC-CM2-SR5":20,
    "MPI-ESM1-2-HR":10,
    "CanESM5":20,
    # "MPI-ESM1-2-LR":10,
    # "CNRM-ESM2-1":20,
    # "MIROC6":20,
}

model_encoding = {
    "NorCPM1":1,
    "EC-Earth3":2,
    "HadGEM3-GC31-MM":3,
    "CMCC-CM2-SR5":4,
    "MPI-ESM1-2-HR":5,
    "CanESM5":6,
    # "MPI-ESM1-2-LR":7,
    # "CNRM-ESM2-1":8,
    # "MIROC6":9,
}

model_decoding = {val:key for key,val in model_encoding.items()}

ensemble_member_dimension = {
    "norcpm1":"mem",
    "ec_earth3":"realization",
    "hadgem3_gc31_mm":"realization",
    "cmcc_cm2_sr5":"realization",
    "mpi_esm1_2_hr":"realization",
    "canesm5":"realization",
    # "mpi_esm1_2_lr":"realization",
    # 'cnrm_esm2_1':'realization',
    # 'miroc6':'realization',
}


