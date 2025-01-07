import subprocess as sbp
from pathlib import Path
from decadal_predictions.config import *
from joblib import Parallel, delayed

VARS = list(var_name_map['cmip_to_cf'].keys())

example_grid_origin_file='/projects/NS9873K/DATA/SFE/ERA5/res_6hrly_1/2m_temperature/2m_temperature_2000_01.nc'

# dpath = data_paths['hindcast']['NorCPM1']
dpath = Path('/projects/NS9873K/owul/data/statkraft_pilot4/decadal_predictions/NorCPM1/dcppA-hindcast/')
all_fi = sorted(list(dpath.rglob('*Amon*.nc')))

rel_vars_files = [fi for fi in all_fi if str(fi.name).split('_')[0] in VARS]

def regrid(infile):
    new_path = Path('/datalake/NS9873K/DATA/DCPP/dcppA-hindcast/NorCPM1_rgr')
    outfile = new_path/infile.name
    if not outfile.exists():
        interp = sbp.run(
            [
                'ncremap',
                '-a',
                'bilin',
                '-d',
                example_grid_origin_file,
                infile,
                outfile
            ]
        )
        if interp.returncode == 0:
            print('Successfully interpolated {}'.format(str(infile)))
    else:
        print('{} already exists, skipping interpolation.'.format(str(outfile)))

fits = Parallel(n_jobs=5)(delayed(regrid)(arr) for arr in rel_vars_files); # ~ 10s
