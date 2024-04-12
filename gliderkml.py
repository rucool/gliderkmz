#!/usr/bin/env python

"""
Author: lgarzio and lnazzaro on 2/28/2024
Last modified: lgarzio on 4/12/2024
Test glider kmz generation
"""

import os
import argparse
import sys
import re
import pytz
from dateutil import parser
import datetime as dt
import pandas as pd
import numpy as np
import requests
import simplekml
import yaml
from jinja2 import Environment, FileSystemLoader
from oceans.ocfis import uv2spdir
from magnetic_field_calculator import MagneticFieldCalculator
pd.set_option('display.width', 320, "display.max_columns", 10)


def add_sensor_values(data_dict, sensor_name, sdf, thresholds=None):
    """
    Find data from a sensor within a specific time range (-5 minutes from surface connect time thru +5 minutes from
    surface disconnect time). Add the median of the values to the dictionary summaries
    """
    if thresholds:
        try:
            sthresholds = thresholds[sensor_name]
        except KeyError:
            sthresholds = None
    else:
        sthresholds = None

    cts = pd.to_datetime(data_dict['connect_ts'])
    dcts = pd.to_datetime(data_dict['disconnect_ts'])
    t0 = cts - pd.Timedelta(minutes=5)
    t1 = dcts + pd.Timedelta(minutes=5)

    try:
        sensor_value = np.round(np.median(sdf.loc[np.logical_and(sdf.ts >= t0, sdf.ts <= t1)].value), 2)
        if np.isnan(sensor_value):
            sensor_value = None
            bgcolor = 'BEA60E'  # yellow BEA60E
        else:
            if sthresholds:
                bgcolor = 'green'
                if 'suspect_low' in sthresholds.keys() and sensor_value <= sthresholds['suspect_low']:
                    bgcolor = 'BEA60E'  # yellow BEA60E
                elif 'suspect_high' in sthresholds.keys() and sensor_value <= sthresholds['suspect_high']:
                    bgcolor = 'BEA60E'  # yellow BEA60E
                elif 'fail_low' in sthresholds.keys() and sensor_value <= sthresholds['fail_low']:
                    bgcolor = 'darkred'  # yellow BEA60E
                elif 'fail_high' in sthresholds.keys() and sensor_value <= sthresholds['fail_high']:
                    bgcolor = 'darkred'  # yellow BEA60E
            else:
                bgcolor = 'BEA60E'  # yellow BEA60E
    except IndexError:
        sensor_value = None
        bgcolor = 'BEA60E'  # yellow BEA60E
    data_dict[sensor_name] = sensor_value
    data_dict[f'{sensor_name}_bgcolor'] = bgcolor


def build_popup_dict(data):
    """
    Build the dictionaries for the data that populates the pop-up text boxes
    :param data: dictionary
    """
    connect_ts = format_ts_epoch(data['connect_time_epoch'])
    disconnect_ts = format_ts_epoch(data['disconnect_time_epoch'])
    gps_connect_ts = format_ts_epoch(data['gps_timestamp_epoch'])

    gps_connect_timedelta = dt.datetime.fromtimestamp(data['connect_time_epoch'], dt.UTC) - dt.datetime.fromtimestamp(data['gps_timestamp_epoch'], dt.UTC)
    if gps_connect_timedelta.seconds >= 30*60:  # 30 minutes (per Dave slack message 4/1/2024)
        gps_bgcolor = 'darkred'
    # elif gps_connect_timedelta.seconds > 10*6:  # between 10 mins and fail (above)
    #     gps_bgcolor = 'BEA60E'  # yellow BEA60E
    else:  # < suspect (or fail if not using suspect range)
        gps_bgcolor = 'green'

    try:
        waypoint_range_km = data['waypoint_range_meters'] / 1000
    except TypeError:
        waypoint_range_km = None

    popup_dict = dict(
        connect_ts=connect_ts,
        disconnect_ts=disconnect_ts,
        gps_lat=np.round(convert_nmea_degrees(data['gps_lat']), 2),
        gps_lon=np.round(convert_nmea_degrees(data['gps_lon']), 2),
        gps_connect_ts=gps_connect_ts,
        gps_bgcolor=gps_bgcolor,
        reason=data['surface_reason'],
        mission=data['mission'],
        filename=data['filename'],
        filename_8x3=data['the8x3_filename'],
        dsvr_log=data['dsvr_log_name'],
        segment_ewo=f"{data['segment_errors']}/{data['segment_warnings']}/{data['segment_oddities']}",
        mission_ewo=f"{data['mission_errors']}/{data['mission_warnings']}/{data['mission_oddities']}",
        total_ewo=f"{data['total_errors']}/{data['total_warnings']}/{data['total_oddities']}",
        waypoint_lat=data['waypoint_lat'],
        waypoint_lon=data['waypoint_lon'],
        waypoint_range=waypoint_range_km,
        waypoint_bearing=data['waypoint_bearing_degrees']
    )

    return popup_dict


def convert_nmea_degrees(x):
    """
    Convert lat/lon coordinates from nmea to decimal degrees
    """
    try:
        degrees = np.sign(x) * (np.floor(np.abs(x)/100) + np.mod(np.abs(x), 100) / 60)
    except TypeError:
        degrees = None

    return degrees


def format_ts_epoch(timestamp):
    return dt.datetime.fromtimestamp(timestamp, dt.UTC).strftime('%Y-%m-%d %H:%M')


def main(args):
    loglevel = args.loglevel.upper()  # TODO do something with logging?
    deployment = args.deployment
    kml_type = args.kml_type

    sensor_list = ['m_battery', 'm_vacuum', 'm_water_vx', 'm_water_vy', 'm_gps_mag_var']

    sensor_thresholds_yml = '/Users/garzio/Documents/repo/lgarzio/gliderkmz/configs/sensor_thresholds.yml'
    with open(sensor_thresholds_yml) as f:
        sensor_thresholds = yaml.safe_load(f)

    templatedir = '/Users/garzio/Documents/repo/rucool/gliderkmz/templates/'

    savedir = '/Users/garzio/Documents/repo/rucool/gliderkmz/files/'
    # savedir = '/www/web/rucool/gliders/kmz'

    glider_tails = 'https://rucool.marine.rutgers.edu/gliders/glider_tails/'  # /www/web/rucool/gliders/glider_tails
    # old glider tails location: https://marine.rutgers.edu/~kerfoot/icons/glider_tails/

    # inspired by colorblind-friendly colormap (https://mpetroff.net/2018/03/color-cycle-picker/) for tracks/points
    # NOTE: kml colors are encoded backwards from the HTML convention. HTML colors are "#rrggbbaa": Red Green Blue
    # Alpha, while KML colors are "AABBGGRR": Alpha Blue Green Red.

    # teal ('ffe9d043'), pink ('ff9e36d7'), purple ('ffd7369e'), yellow ('ff43d0e9'), orange ('ff3877f3'),
    # green ('ff83c995'), gray ('ffc4c9d8')
    colors = ['ffe9d043', 'ff9e36d7', 'ffd7369e', 'ff43d0e9', 'ff3877f3', 'ff83c995', 'ffc4c9d8']

    # load the templates
    environment = Environment(loader=FileSystemLoader(templatedir))
    template = environment.get_template('kml_template.kml')
    format_template = environment.get_template('format_active_deployments_macro.kml')
    deployment_template = environment.get_template('deployment_macro.kml')
    track_template = environment.get_template('track_macro.kml')
    surfacing_template = environment.get_template('surface_event_macro.kml')
    text_box_template = environment.get_template('text_box_macro.kml')

    # define filename
    if kml_type == 'deployed':
        ext = ''
    else:
        ext = f'_{kml_type.split("deployed_")[-1]}'

    ts_now = dt.datetime.now(dt.UTC).strftime('%m/%d/%y %H:%M')

    glider_api = 'https://marine.rutgers.edu/cool/data/gliders/api/'

    glider_deployments = []
    if deployment == 'active':  # 'deployed' 'deployed_ts' 'deployed_uv' 'deployed_ts_uv'
        savefile = os.path.join(savedir, f'active_deployments{ext}.kml')
        document_name = 'Active Deployments'
        active_deployments = requests.get(f'{glider_api}deployments/?active').json()['data']
        for ad in active_deployments:
            glider_deployments.append(ad['deployment_name'])

        # duplicate track colors if necessary
        if len(glider_deployments) > len(colors):
            repeatx = int(np.ceil(len(glider_deployments) / len(colors)))
            colors = colors * repeatx

    else:
        glider_regex = re.compile(r'^(.*)-(\d{8}T\d{4})')
        match = glider_regex.search(deployment)
        if match:
            try:
                (glider, trajectory) = match.groups()
                try:
                    trajectory_dt = parser.parse(trajectory).replace(tzinfo=pytz.UTC)
                except ValueError as e:
                    # logger.error('Error parsing trajectory date {:s}: {:}'.format(trajectory, e))
                    print('need to enable logging')
            except ValueError as e:
                # logger.error('Error parsing invalid deployment name {:s}: {:}'.format(deployment, e))
                print('need to enable logging')
        else:
            # logger.error('Cannot pull glider name from {:}'.format(deployment))
            print('need to enable logging')
            ## exit the script????

        savedir = os.path.join(savedir, 'deployments', str(trajectory_dt.year), deployment)
        os.makedirs(savedir, exist_ok=True)
        savefile = os.path.join(savedir, f'{deployment}{ext}.kml')
        document_name = 'Glider Deployments'
        glider_deployments.append(deployment)
        colors = ['ff43d0e9']  # yellow

    format_dict = dict()
    deployment_dict = dict()
    for idx, gd in enumerate(glider_deployments):
        deployment_api = requests.get(f'{glider_api}deployments/?deployment={gd}').json()['data'][0]

        # build the dictionary for the formatting section of the kml
        glider_name = deployment_api['glider_name']
        deployment_name = deployment_api['deployment_name']
        glider_tail = os.path.join(glider_tails, f'{glider_name}.png')
        format_dict[deployment_name] = dict(
            name=glider_name,
            glider_tail=glider_tail,
            deployment_color=colors[idx]
        )

        # get distance flown and calculate days deployed
        distance_flown_km = deployment_api['distance_flown_km']
        try:
            end = dt.datetime.fromtimestamp(deployment_api['end_date_epoch'], dt.UTC)
        except TypeError:
            end = dt.datetime.now(dt.UTC)
        start = dt.datetime.fromtimestamp(deployment_api['start_date_epoch'], dt.UTC)
        seconds_deployed = ((end - start).days * 86400) + (end - start).seconds
        days_deployed = np.round(seconds_deployed / 86400, 2)

        # grab the data from the surface sensors and store in a dictionary (so you only have to hit the API once
        # per sensor per deployment)
        sensor_data = dict()
        for sensor in sensor_list:
            sensor_api = requests.get(f'{glider_api}sensors/?deployment={deployment_name}&sensor={sensor}').json()['data']
            if len(sensor_api) > 0:
                sensor_df = pd.DataFrame(sensor_api)
                sensor_df.sort_values(by='epoch_seconds', inplace=True, ignore_index=True)
                sensor_df['ts'] = pd.to_datetime(sensor_df['ts'])
                if sensor == 'm_gps_mag_var':
                    sensor_df['value'] = sensor_df['value'] * 180 / np.pi
                    sensor_df['units'] = 'degrees'
                sensor_data[sensor] = sensor_df
        
        # add m_gps_mag_var proxy if not being sent back
        if 'm_water_vx' in sensor_data.keys() and 'm_water_vy' in sensor_data.keys() and 'm_gps_mag_var' not in sensor_data.keys():
            calculator = MagneticFieldCalculator()
            sensor_df = sensor_data['m_water_vx'][['ts', 'epoch_seconds', 'lat', 'lon']].copy()
            sensor_df.insert(0, 'sensor', 'calculated_declination')
            # sensor_df.insert(1, 'units', 'rad')
            sensor_df.insert(1, 'units', 'degrees')
            sensor_df.insert(2, 'value', np.nan)
            sensor_df['date'] = sensor_df['ts'].dt.date
            for d in np.unique(sensor_df['date']):
                di = sensor_df['date'] == d
                result = calculator.calculate(latitude=np.nanmedian(sensor_df['lat'][di]),
                                longitude=np.nanmedian(sensor_df['lon'][di]),
                                altitude=0,
                                date=d)
                # sensor_df.loc[di,'value'] = -result['field-value']['declination']['value']*np.pi/180
                sensor_df.loc[di, 'value'] = -result['field-value']['declination']['value']  # units = degrees
            sensor_data['m_gps_mag_var'] = sensor_df

        # track information
        # gather track timestamp and location from the API
        track_dict = dict(
            gps_epoch=np.array([], dtype='int'),
            lon=np.array([], dtype='float'),
            lat=np.array([], dtype='float'),
            sid=np.array([], dtype='int')
        )
        track_features = requests.get(f'{glider_api}tracks/?deployment={deployment_name}').json()['features']
        for tf in track_features:
            if tf['geometry']['type'] == 'Point':
                track_dict['gps_epoch'] = np.append(track_dict['gps_epoch'], tf['properties']['gps_epoch'])
                track_dict['lon'] = np.append(track_dict['lon'], tf['geometry']['coordinates'][0])
                track_dict['lat'] = np.append(track_dict['lat'], tf['geometry']['coordinates'][1])
                track_dict['sid'] = np.append(track_dict['sid'], tf['properties']['sid'])

        # add the last surfacing to the dictionary
        ls_api = deployment_api['last_surfacing']
        track_dict['gps_epoch'] = np.append(track_dict['gps_epoch'], ls_api['connect_time_epoch'])
        track_dict['lon'] = np.append(track_dict['lon'], ls_api['gps_lon_degrees'])
        track_dict['lat'] = np.append(track_dict['lat'], ls_api['gps_lat_degrees'])
        track_dict['sid'] = np.append(track_dict['sid'], ls_api['surfacing_id'])

        # convert to dataframe to sort by time
        track_df = pd.DataFrame(track_dict)
        track_df.sort_values(by='gps_epoch', inplace=True, ignore_index=True)

        if kml_type in ['deployed', 'deployed_uv']:
            track_df = track_df.copy()[['lon', 'lat']]
            track_df['height'] = 4.999999999999999
            track_values = track_df.values.tolist()
            kml = simplekml.Kml()
            track_data = kml.newlinestring(name="track")
            for values in track_values:
                track_data.coords.addcoordinates([(values[0], values[1], values[2])])
        elif kml_type in ['deployed_ts', 'deployed_ts_uv']:
            # build the dictionary that contains the track information to input into the kml template
            track_data = dict()
            for idx, row in track_df.iterrows():
                if idx > 0:
                    prev_row = track_df.iloc[idx - 1]
                    start = dt.datetime.fromtimestamp(prev_row.gps_epoch, dt.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')
                    end = dt.datetime.fromtimestamp(row.gps_epoch, dt.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')
                    track_data[idx] = dict(
                        start=start,
                        end=end,
                        start_lon=prev_row.lon,
                        start_lat=prev_row.lat,
                        end_lon=row.lon,
                        end_lat=row.lat
                    )

        # surface events
        surface_events = requests.get(f'{glider_api}surfacings/?deployment={deployment_name}').json()['data']
        surface_events_df = pd.DataFrame(surface_events)
        surface_events_df.sort_values(by='connect_time_epoch', inplace=True, ignore_index=True)

        # calculate previous 24 hours
        t24h = pd.to_datetime(ts_now) - pd.Timedelta(hours=24)

        surface_events_dict = dict()
        currents_dict = dict()
        call_length_seconds = 0

        # build the information for the surfacings and depth-averaged currents
        for idx, row in surface_events_df.iterrows():
            se = row.to_dict()

            call_length_seconds = call_length_seconds + se['call_length_seconds']
            surface_event_popup = build_popup_dict(se)

            # define surfacing grouping (e.g. last 24 hours or day)
            se_ts = pd.to_datetime(surface_event_popup['connect_ts'])

            if se_ts >= t24h:
                folder_name = 'Last 24 Hours'
                style_name = 'RecentSurfacing'
            else:
                folder_name = se_ts.strftime('%Y-%m-%d')
                style_name = 'Surfacing'

            # define folder name for depth-average currents
            currents_folder_name = se_ts.strftime('%Y-%m-%d')
            connect_datetime = dt.datetime.fromtimestamp(se['connect_time_epoch'], dt.UTC)

            # add the folder name to the surface events dictionary if it's not already there
            try:
                surface_events_dict[folder_name]
            except KeyError:
                surface_events_dict[folder_name] = dict()

            # add the folder name to the currents dictionary if it's not already there
            try:
                currents_dict[currents_folder_name]
            except KeyError:
                currents_dict[currents_folder_name] = dict()

            # build dictionary for depth-averaged currents
            currents_dict[currents_folder_name][idx] = dict(
                connect_HHMM=connect_datetime.strftime('%H:%M'),
                connect_ts_Z=connect_datetime.strftime('%Y-%m-%dT%H:%M:%SZ'),
                connect_ts=surface_event_popup['connect_ts'],
                disconnect_ts=surface_event_popup['disconnect_ts'],
                lon_degrees_start=se['gps_lon_degrees'],
                lat_degrees_start=se['gps_lat_degrees']
            )

            # calculate depth-averaged currents
            # find m_water_vx, m_water_vy and m_gps_mag_var for this surfacing
            for sensor in ['m_water_vx', 'm_water_vy', 'm_gps_mag_var']:
                add_sensor_values(currents_dict[currents_folder_name][idx], sensor, sensor_data[sensor])

            # rotate from magnetic plane to true plane and calculate current speed and angle
            surfacing_m_water_vx = currents_dict[currents_folder_name][idx]['m_water_vx']
            surfacing_m_water_vy = currents_dict[currents_folder_name][idx]['m_water_vy']
            surfacing_m_gps_mag_var = currents_dict[currents_folder_name][idx]['m_gps_mag_var']

            if np.logical_and(np.logical_and(surfacing_m_water_vx, surfacing_m_water_vy), surfacing_m_gps_mag_var):
                # units: current_angle = degrees, current_speed = m/s
                current_angle, current_speed = uv2spdir(surfacing_m_water_vx, surfacing_m_water_vy,
                                                        mag=-surfacing_m_gps_mag_var)
            elif np.logical_and(surfacing_m_water_vx, surfacing_m_water_vy):
                # calculate m_gps_mag_var if it's not there
                calculator = MagneticFieldCalculator()
                result = calculator.calculate(latitude=se['gps_lat_degrees'],
                                              longitude=se['gps_lon_degrees'],
                                              altitude=0,
                                              date=currents_folder_name)
                mag = -result['field-value']['declination']['value']
                current_angle, current_speed = uv2spdir(surfacing_m_water_vx, surfacing_m_water_vy, mag=-mag)
            else:
                current_angle = None
                current_speed = None

            # TODO: calculate where the glider will be in 1 day floating at surface and replace lon_deg_end and lat_deg_end with that
            lon_deg_end = se['gps_lon_degrees'] - .05  # these are placeholders
            lat_deg_end = se['gps_lat_degrees'] - .05  # these are placeholders

            currents_dict[currents_folder_name][idx]['lon_degrees_end'] = lon_deg_end
            currents_dict[currents_folder_name][idx]['lat_degrees_end'] = lat_deg_end

            surface_events_dict[folder_name][idx] = dict(
                connect_ts=surface_event_popup['connect_ts'],
                connect_ts_Z=connect_datetime.strftime('%Y-%m-%dT%H:%M:%SZ'),
                gps_lat_degrees=se['gps_lat_degrees'],
                gps_lon_degrees=se['gps_lon_degrees'],
                style_name=style_name,
                surface_event_popup=surface_event_popup
            )

            # add values for battery and vacuum to the surface event popup
            for sensor in ['m_battery', 'm_vacuum']:
                add_sensor_values(surface_events_dict[folder_name][idx]['surface_event_popup'],
                                  sensor, sensor_data[sensor], thresholds=sensor_thresholds)

            # TODO calculate dive/speed information
            # add dive and current information to the surfacing event (time, distance, speed)
            surface_events_dict[folder_name][idx]['surface_event_popup']['dive_time'] = None  # minutes
            surface_events_dict[folder_name][idx]['surface_event_popup']['dive_dist'] = None  # km
            surface_events_dict[folder_name][idx]['surface_event_popup']['total_speed'] = None  # m/s
            surface_events_dict[folder_name][idx]['surface_event_popup']['total_speed_bearing'] = None
            surface_events_dict[folder_name][idx]['surface_event_popup']['current_speed'] = current_speed  # m/s
            surface_events_dict[folder_name][idx]['surface_event_popup']['current_speed_bearing'] = current_angle
            surface_events_dict[folder_name][idx]['surface_event_popup']['glide_speed'] = None  # m/s
            surface_events_dict[folder_name][idx]['surface_event_popup']['glide_speed_bearing'] = None

            if idx == 0:  # deployment location
                deployment_popup_dict = surface_events_dict[folder_name][idx]['surface_event_popup']
                deployment_ts_Z = dt.datetime.fromtimestamp(se['connect_time_epoch'], dt.UTC).strftime(
                    '%Y-%m-%dT%H:%M:%SZ')
                deployment_gps_lat_degrees = se['gps_lat_degrees']
                deployment_gps_lon_degrees = se['gps_lon_degrees']

            if idx == len(surface_events_df) - 1:  # last surfacing
                last_surfacing_popup_dict = surface_events_dict[folder_name][idx]['surface_event_popup']
                ls_gps_lat_degrees = se['gps_lat_degrees']
                ls_gps_lon_degrees = se['gps_lon_degrees']

        deployment_dict[deployment_name] = dict(
            ts_now=ts_now,
            glider_name=glider_name,
            glider_tail=glider_tail,
            ls_connect_ts=last_surfacing_popup_dict['connect_ts'],
            deploy_ts_Z=deployment_ts_Z,
            ls_gps_lat_degrees=ls_gps_lat_degrees,
            ls_gps_lon_degrees=ls_gps_lon_degrees,
            last_surfacing_popup=last_surfacing_popup_dict,
            deploy_connect_ts=deployment_popup_dict['connect_ts'],
            deploy_gps_lat_degrees=deployment_gps_lat_degrees,
            deploy_gps_lon_degrees=deployment_gps_lon_degrees,
            deployment_popup=deployment_popup_dict,
            cwpt_since=last_surfacing_popup_dict['disconnect_ts'],
            cwpt_lat=ls_api['waypoint_lat'],
            cwpt_lon=ls_api['waypoint_lon'],
            cwpt_lat_degrees=convert_nmea_degrees(ls_api['waypoint_lat']),
            cwpt_lon_degrees=convert_nmea_degrees(ls_api['waypoint_lon']),
            distance_flown_km=distance_flown_km,
            days_deployed=days_deployed,
            iridium_mins=int(np.round(call_length_seconds / 60)),
            track_info=track_data,
            surface_event_info=surface_events_dict,
            currents_info=currents_dict
        )

    # render all of the information into the kml template
    content = template.render(
        document_name=document_name,
        kml_type=kml_type,
        format_info=format_dict,
        deployment_info=deployment_dict
    )

    with open(savefile, mode="w", encoding="utf-8") as message:
        message.write(content)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=main.__doc__,
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('deployment',
                            type=str,
                            help='Use "active" to generate one kml file for all active deployments, or provide '
                                 'one glider deployment name formatted as glider-YYYYmmddTHHMM')

    arg_parser.add_argument('-kml_type',
                            type=str,
                            help='Type to generate. "deployed": glider track with surfacings, "deployed_ts": same as'
                                 'deployed with the option to filter by time, "deployed_uv": same as deployed with '
                                 'vectors of depth-averaged currents, and "deployed_ts_uv": same as deployed_ts with'
                                 'vectors of depth-averaged currents.',
                            choices=['deployed', 'deployed_ts', 'deployed_uv', 'deployed_ts_uv'],
                            default='deployed')

    arg_parser.add_argument('-l', '--loglevel',
                            help='Verbosity level',
                            type=str,
                            choices=['debug', 'info', 'warning', 'error', 'critical'],
                            default='info')

    parsed_args = arg_parser.parse_args()

    sys.exit(main(parsed_args))
