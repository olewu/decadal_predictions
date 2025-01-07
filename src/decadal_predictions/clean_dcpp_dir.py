# do a search through all directories in /datalake/NS9873K/DATA/DCPP/dcppA-hindcast/ and find all .nc files that have two eight digit sequences separated by a "-" and delete those if there exists another file with the same start year indicated by sYYYY
from decadal_predictions.config import *

# find all files that should be removed:
no_use_files = sorted(list(data_paths['hindcast']['EC-Earth3'].parent.rglob('*r10*_????????-????????.nc')))

# but keep those where no alternative data exists:
rem = []
norem = []
for nuf in no_use_files:
    spatt = ''.join(str(nuf.name).split('r10')[:-1])
    keep_files = sorted(list(nuf.parent.glob(f'{spatt}*_??????-??????.nc')))
    if len(keep_files) > 0:
        # remove nuf:
        nuf.unlink()
        print('\tremoving {}'.format(str(nuf)))
        rem.append(str(nuf))
    else:
        norem.append(str(nuf))
        print('not removing {}, no alternative exists.'.format(str(nuf)))

# len([nr for nr in norem if 'MPI-ESM1-2-LR' not in nr])


# ncpm_wind_amon = sorted(list(Path('/datalake/NS9873K/DATA/DCPP/dcppA-hindcast/NorCPM1_rgr').glob('sfcWind*????????-????????.nc')))
# for ncpm_file in ncpm_wind_amon:
#     sdates = re.search(r'(\d{8})-(\d{8})$',str(ncpm_file.stem))
#     target = Path(str(ncpm_file).replace(sdates.group(1),sdates.group(1)[:-2]).replace(sdates.group(2),sdates.group(2)[:-2]))
#     _ = ncpm_file.replace(target)