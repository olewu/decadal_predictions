import subprocess as sbp

from decadal_predictions.config import *

import requests
import sys
import re

# node: "https://esgf-metagrid.cloud.dkrz.de/search" does not work like this

def esgf_search(server="https://esgf-node.llnl.gov/esg-search/search",
                files_type="OPENDAP", local_node=True, project="CMIP6",
                verbose=False, format="application%2Fsolr%2Bjson",
                use_csrf=False, **search):
    client = requests.session()
    payload = search
    payload["project"] = project
    payload["type"]= "File"
    if local_node:
        payload["distrib"] = "false"
    if use_csrf:
        client.get(server)
        if 'csrftoken' in client.cookies:
            # Django 1.6 and up
            csrftoken = client.cookies['csrftoken']
        else:
            # older versions
            csrftoken = client.cookies['csrf']
        payload["csrfmiddlewaretoken"] = csrftoken

    payload["format"] = format

    offset = 0
    numFound = 10000
    all_files = []
    files_type = files_type.upper()
    while offset < numFound:
        payload["offset"] = offset
        url_keys = []
        for k in payload:
            url_keys += ["{}={}".format(k, payload[k])]

        url = "{}/?{}".format(server, "&".join(url_keys))
        print(url)
        r = client.get(url)
        r.raise_for_status()
        resp = r.json()["response"]
        numFound = int(resp["numFound"])
        resp = resp["docs"]
        offset += len(resp)
        for d in resp:
            if verbose:
                for k in d:
                    print("{}: {}".format(k,d[k]))
            url = d["url"]
            for f in d["url"]:
                sp = f.split("|")
                if sp[-1] == files_type:
                    all_files.append(sp[0].split(".html")[0])
    return sorted(all_files)

def download_dcpp_simulations(MODEL,variables,syears,exp='dcppA-hindcast',overwrite=False,example_grid_origin_file=example_grid_file):

    for variable in variables:
        for iyear in syears:
            result = esgf_search(
                activity_id='DCPP',
                table_id='Amon',
                variable_id=variable,
                experiment_id=exp,
                source_id=MODEL,
                sub_experiment_id=f's{iyear}',
                latest=True # get only latest version
            )

            if not result:

                print(f'no results for {variable} {exp} {MODEL} {iyear} in DCPP Amon')

                continue

            # catch cases where the same simulations are hosted on several nodes:
            
            if len(list(set([res.split('/')[2] for res in result]))) > 1:
                result = list({Path(res).name: res for res in result}.values())

            # group results by single member
            members = sorted(list(set([re.search(r'r\d+i\d+p\d+f\d+',res).group(0) for res in result])))

            for member in members:

                member_list = sorted([res for res in result if member in res])

                end_dates = sorted([re.search(r'-(\d+).nc',mem).group(1) for mem in member_list])

                out_merge_name = str(Path(member_list[0]).name).replace(end_dates[0],end_dates[-1])

                out_merge_full = str(data_paths['hindcast'][MODEL]/out_merge_name)
                data_paths['hindcast'][MODEL].mkdir(parents=True,exist_ok=True)
                Path(str(data_paths['hindcast'][MODEL])+'_rgr').mkdir(parents=True,exist_ok=True)

                if (not Path(out_merge_full).exists()) or (overwrite):
                    if len(member_list) > 0:
                        # open files and concatenate
                        cat = sbp.run(
                            [
                                'ncrcat',
                                '-O',
                                '-h',
                                *member_list,
                                out_merge_full
                            ]
                        )
                        ret = cat.returncode
                    else:
                        print(f'no members found for {out_merge_name}')


                    if ret == 0:
                        out_interp_full = out_merge_full.replace(f'/{MODEL}/',f'/{MODEL}_rgr/')
                        out_interp_name = str(Path(out_interp_full).name)
                        interp = sbp.run(
                            [
                                'ncremap',
                                '-a',
                                'bilin',
                                '-d',
                                example_grid_origin_file,
                                out_merge_full,
                                out_interp_full
                            ]
                        )
                        if interp.returncode != 0:
                            print(f'regridding failed on {out_interp_full}')
                        else:
                            print(f'\tSuccessfully downloaded and regridded {out_merge_name}')

                    else:
                        print(f'Could not download/concatenate {out_merge_name}')

if __name__ == '__main__':
    
    # total arguments
    n = len(sys.argv)
    print("Total arguments passed:", n)

    # Arguments passed
    print("\nName of Python script:", sys.argv[0])

    MODEL = sys.argv[1]
    variables = [sys.argv[2]] # ['psl','sfcWind','rsds','tas','pr']
    syears = range(int(sys.argv[3]),int(sys.argv[4]))
    download_dcpp_simulations(MODEL,variables,syears)