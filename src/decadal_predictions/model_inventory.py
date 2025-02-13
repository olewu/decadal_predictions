from decadal_predictions.config import *
from decadal_predictions.download import *
import subprocess as sbp
import re

# for each model and variable print the number of ensemble members per init year:
for model in available_models_long:

    spath = Path(str(data_paths['hindcast'][model]) + '_rgr')

    all_files = sorted(list(spath.rglob('*.nc')))

    for variable in var_name_map['cmip_to_cf'].keys():

        # if variable != 'psl':
        #     continue

        all_files_for_var = [fi for fi in all_files if fi.stem.split('_')[0] == variable]
        syears = []
        for year in range(1960,2019):
            all_files_for_var_for_year = [fi for fi in all_files_for_var if int(re.search(r'_s(\d{4})-',fi.stem).group(1)) == year]

            ens_code = [re.search(r'-(r\d{1,2}i\d{1,2}p\d{1,2}f\d{1,2})_',fi.stem).group(1) for fi in all_files_for_var_for_year]

            n_unique_members = len(set(ens_code))

            # print(f'{model} {variable} {year}: {n_unique_members}')

            if n_unique_members < expected_hindcast_ens_size[model]:
                # print('collecting missing years')
                syears.append(year)
                print(f'{model} {variable} {year}: {n_unique_members}')

            if len(ens_code) != n_unique_members:

                duplicates = [em for em in set(ens_code) if ens_code.count(em) > 1]

                print('\tduplicate members: {}'.format(duplicates))

        if syears:
            print(model,variable)
            print(syears)
            print('\tdownloading...')
            download_dcpp_simulations(model,[variable],syears)

# fix missing re-gridding:
# for model in available_models_long:
#     print(model)
#     non_reg_path = data_paths['hindcast'][model]
#     reg_path = Path(str(data_paths['hindcast'][model]) + '_rgr')

#     all_nonreg = non_reg_path.glob('*.nc')
#     all_nonreg_files = sorted([fi.stem for fi in all_nonreg])

#     all_reg = reg_path.glob('*.nc')
#     all_reg_files = sorted([fi.stem for fi in all_reg])

#     missed_files = [fi for fi in all_nonreg_files if fi not in all_reg_files]

#     print(len(missed_files))
#     print(missed_files)
    
#     for mfile in missed_files:
#         in_noninterp = str(non_reg_path/(mfile+'.nc'))
#         out_interp_full = str(reg_path/(mfile+'.nc'))
#         interp = sbp.run(
#             [
#                 'ncremap',
#                 '-a',
#                 'bilin',
#                 '-d',
#                 example_grid_file,
#                 in_noninterp,
#                 out_interp_full
#             ]
#         )

#         if interp.returncode != 0:
#             print(f'failed regridding {mfile}')

