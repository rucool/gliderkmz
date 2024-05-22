#!/usr/bin/env python

"""
Author: lgarzio and lnazzaro on 5/21/2024
Last modified: lgarzio on 5/22/2024
Generate a .yaml file of active deployments and associated track color
"""

import os
import argparse
import sys
import datetime as dt
import pandas as pd
import numpy as np
import requests
import yaml

pd.set_option('display.width', 320, "display.max_columns", 10)


def main(args):
    configdir = args.configdir

    # inspired by colorblind-friendly colormap (https://mpetroff.net/2018/03/color-cycle-picker/) for tracks/points
    # NOTE: kml colors are encoded backwards from the HTML convention. HTML colors are "#rrggbbaa": Red Green Blue
    # Alpha, while KML colors are "AABBGGRR": Alpha Blue Green Red.

    # teal ('ffe9d043'), purple ('ffd7369e'), yellow ('ff43d0e9'), pink ('ff9e36d7'), orange ('ff3877f3'),
    # green ('ff83c995'), gray ('ffc4c9d8')
    colors = ['ffe9d043', 'ffd7369e', 'ff43d0e9', 'ff9e36d7', 'ff3877f3', 'ff83c995', 'ffc4c9d8']

    ts_now = dt.datetime.now(dt.UTC).strftime('%Y%m%dT%H%M')

    # get the current active deployments from the API
    glider_api = 'https://marine.rutgers.edu/cool/data/gliders/api/'
    active_deployments_api = requests.get(f'{glider_api}deployments/?active').json()['data']
    glider_deployments = []
    for ad in active_deployments_api:
        glider_deployments.append(ad['deployment_name'])

    savefile = os.path.join(configdir, 'active_deployments.yml')

    # set up the config file for the first time
    if not os.path.isfile(savefile):
        ad_config = dict(
            deployments=dict(),
            updated=ts_now
        )

        # duplicate track colors if necessary
        if len(glider_deployments) > len(colors):
            repeatx = int(np.ceil(len(glider_deployments) / len(colors)))
            colors = colors * repeatx

        for coloridx, gd in enumerate(glider_deployments):
            ad_config['deployments'][gd] = colors[coloridx]

    # if the config file is already there, update it
    else:
        with open(savefile) as f:
            ad_config = yaml.safe_load(f)

        ad_config['updated'] = ts_now

        # remove inactive deployments from config file
        inactive = list(set(list(ad_config['deployments'].keys())).difference(glider_deployments))
        for ia in inactive:
            del ad_config['deployments'][ia]

        # find new active deployments
        new_deployments = list(set(glider_deployments).difference(list(ad_config['deployments'].keys())))
        new_deployments_num = len(new_deployments)
        if new_deployments_num > 0:

            # find the track colors that are already being used
            used_colors = []
            for deploy, tcolor in ad_config['deployments'].items():
                used_colors.append(tcolor)

            # find the colors that haven't been used
            free_colors = list(set(colors).difference(used_colors))

            # duplicate track colors if necessary
            if new_deployments_num > len(free_colors):
                repeatx = int(np.ceil(new_deployments_num / len(colors)))
                colors = colors * repeatx
                free_colors += colors

            # add the new deployment to the config file with a new track color
            for coloridx, nd in enumerate(new_deployments):
                ad_config['deployments'][nd] = free_colors[coloridx]

    with open(savefile, 'w') as outfile:
        yaml.dump(ad_config, outfile, default_flow_style=False)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=main.__doc__,
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('-c', '--configdir',
                            type=str,
                            help='Directory containing active_deployments.yml configuration file (if available), or '
                                 'directory where new active_deployments.yml file will be saved')

    parsed_args = arg_parser.parse_args()

    sys.exit(main(parsed_args))
