#!/usr/bin/env python

"""
Author: lgarzio and lnazzaro on 2/28/2024
Last modified: lgarzio on 5/13/2024
Generate glider .kmzs for either 1) all active deployments or 2) a user specified deployment
"""

import os
import argparse
import sys
import zipfile
import re
import pytz
from dateutil import parser
import datetime as dt
import pandas as pd
import numpy as np
import requests
import simplekml
import yaml
import geopy.distance
import math
from jinja2 import Environment, FileSystemLoader
from oceans.ocfis import uv2spdir, spdir2uv
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
    t0 = cts - pd.Timedelta(minutes=15)

    try:
        sdf_sel = sdf.loc[np.logical_and(sdf.ts >= t0, sdf.ts <= dcts)].sort_values(by='epoch_seconds')
        sensor_value = format_float(np.array(sdf_sel.value)[-1])  # grab the latest value reported for this surfacing
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
                elif 'fail_high' in sthresholds.keys() and sensor_value >= sthresholds['fail_high']:
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

    segment_ewo = f"{format_int(data['segment_errors'])}/{format_int(data['segment_warnings'])}/{format_int(data['segment_oddities'])}"
    mission_ewo = f"{format_int(data['mission_errors'])}/{format_int(data['mission_warnings'])}/{format_int(data['mission_oddities'])}"
    total_ewo = f"{format_int(data['total_errors'])}/{format_int(data['total_warnings'])}/{format_int(data['total_oddities'])}"

    try:
        waypoint_range_km = data['waypoint_range_meters'] / 1000
    except TypeError:
        waypoint_range_km = None

    popup_dict = dict(
        connect_ts=connect_ts,
        disconnect_ts=disconnect_ts,
        gps_lat=format_coordinates(data['gps_lat']),
        gps_lon=format_coordinates(data['gps_lon']),
        gps_connect_ts=gps_connect_ts,
        gps_bgcolor=gps_bgcolor,
        reason=data['surface_reason'],
        mission=data['mission'],
        filename=data['filename'],
        filename_8x3=data['the8x3_filename'],
        dsvr_log=data['dsvr_log_name'],
        segment_ewo=format_ewo(segment_ewo),
        mission_ewo=format_ewo(mission_ewo),
        total_ewo=format_ewo(total_ewo),
        waypoint_lat=format_coordinates(data['waypoint_lat']),
        waypoint_lon=format_coordinates(data['waypoint_lon']),
        waypoint_range=format_float(waypoint_range_km),
        waypoint_bearing=format_int(data['waypoint_bearing_degrees'])
    )

    return popup_dict


def calculate_compass_bearing(pointA, pointB):
    """
    Calculates the bearing between two points.
    From https://gist.github.com/jeromer/2005586?permalink_comment_id=4669453 and
    https://towardsdatascience.com/calculating-the-bearing-between-two-geospatial-coordinates-66203f57e4b4
    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    :Parameters:
      - `pointA: The tuple representing the latitude/longitude for the
        first point. Latitude and longitude must be in decimal degrees
      - `pointB: The tuple representing the latitude/longitude for the
        second point. Latitude and longitude must be in decimal degrees
    :Returns:
      The bearing in degrees
    :Returns Type:
      float
    """
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing


def convert_kml_to_kmz(kml_file_path, kmz_file_path=None):
    if kmz_file_path is None:
        kmz_file_path = f'{os.path.splitext(kml_file_path)[0]}.kmz'
    with zipfile.ZipFile(kmz_file_path, 'w', zipfile.ZIP_DEFLATED) as kmz:
        # Define the arcname to be ‘doc.kml’ as per KMZ file specification
        kmz.write(kml_file_path, arcname='doc.kml')
    return kmz_file_path


def format_coordinates(x, decimal_degrees=False):
    """
    Convert lat/lon coordinates from nmea to degrees decimal minutes
    """
    try:
        decdegrees = np.sign(x) * (np.floor(np.abs(x)/100) + np.mod(np.abs(x), 100) / 60)

        # convert from decimal degrees to degrees decimal minutes
        minutes, degrees = math.modf(decdegrees)
        output = f'{int(degrees)} {np.round(abs(minutes * 60), 3)}'

        if decimal_degrees:  # return decimal degrees instead of degrees decimal minutes
            output = decdegrees

    except TypeError:
        output = None

    return output


def format_ewo(ewo):
    if ewo == 'None/None/None':
        ewo = None

    return ewo


def format_float(value):
    # round float values to 2 decimal places
    try:
        value = np.round(value, 2)
    except (ValueError, TypeError):
        value = value

    return value


def format_int(value):
    try:
        value = int(value)
    except (ValueError, TypeError):
        value = value

    return value


def format_ts_epoch(timestamp):
    return dt.datetime.fromtimestamp(timestamp, dt.UTC).strftime('%Y-%m-%d %H:%M')


def main(args):
    loglevel = args.loglevel.upper()  # TODO do something with logging?
    deployment = args.deployment
    kml_type = args.kml_type
    savedir = args.savedir

    sensor_list = ['m_battery', 'm_vacuum', 'm_water_vx', 'm_water_vy', 'm_gps_mag_var']

    kmz_repo = '/home/glideradm/code/kmz/gliderkmz'
    sensor_thresholds_yml = os.path.join(kmz_repo, 'configs/sensor_thresholds.yml')
    with open(sensor_thresholds_yml) as f:
        sensor_thresholds = yaml.safe_load(f)

    templatedir = os.path.join(kmz_repo, 'templates/')

    glider_tails = 'https://rucool.marine.rutgers.edu/gliders/glider_tails/'  # /www/web/rucool/gliders/glider_tails
    # old glider tails location: https://marine.rutgers.edu/~kerfoot/icons/glider_tails/

    # inspired by colorblind-friendly colormap (https://mpetroff.net/2018/03/color-cycle-picker/) for tracks/points
    # NOTE: kml colors are encoded backwards from the HTML convention. HTML colors are "#rrggbbaa": Red Green Blue
    # Alpha, while KML colors are "AABBGGRR": Alpha Blue Green Red.

    # teal ('ffe9d043'), purple ('ffd7369e'), yellow ('ff43d0e9'), pink ('ff9e36d7'), orange ('ff3877f3'),
    # green ('ff83c995'), gray ('ffc4c9d8')
    colors = ['ffe9d043', 'ffd7369e', 'ff43d0e9', 'ff9e36d7', 'ff3877f3', 'ff83c995', 'ffc4c9d8']

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
    if deployment == 'active':  # 'deployed' 'deployed_ts' 'deployed_uv' 'deployed_uv_ts'
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
    for glider_idx, gd in enumerate(glider_deployments):
        deployment_api = requests.get(f'{glider_api}deployments/?deployment={gd}').json()['data'][0]

        # build the dictionary for the formatting section of the kml
        glider_name = deployment_api['glider_name']
        deployment_name = deployment_api['deployment_name']
        glider_tail = os.path.join(glider_tails, f'{glider_name}.png')
        format_dict[deployment_name] = dict(
            name=glider_name,
            glider_tail=glider_tail,
            deployment_color=colors[glider_idx]
        )

        # get distance flown and calculate days deployed
        distance_flown_km = deployment_api['distance_flown_km']
        try:
            end = dt.datetime.fromtimestamp(deployment_api['end_date_epoch'], dt.UTC)
        except TypeError:
            end = dt.datetime.now(dt.UTC)
        start = dt.datetime.fromtimestamp(deployment_api['start_date_epoch'], dt.UTC)
        seconds_deployed = ((end - start).days * 86400) + (end - start).seconds
        days_deployed = format_float(seconds_deployed / 86400)

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
            calculator = MagneticFieldCalculator(model='igrf')
            sensor_df = sensor_data['m_water_vx'][['ts', 'epoch_seconds', 'lat', 'lon']].copy()
            sensor_df.insert(0, 'sensor', 'calculated_declination')
            sensor_df.insert(1, 'units', 'degrees')
            sensor_df.insert(2, 'value', np.nan)
            sensor_df['date'] = sensor_df['ts'].dt.date
            for d in np.unique(sensor_df['date']):
                di = sensor_df['date'] == d
                result = calculator.calculate(latitude=np.nanmedian(sensor_df['lat'][di]),
                                longitude=np.nanmedian(sensor_df['lon'][di]),
                                altitude=0,
                                date=d)
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
        elif kml_type in ['deployed_ts', 'deployed_uv_ts']:
            # build the dictionary that contains the track information to input into the kml template
            track_data = dict()
            for track_idx, row in track_df.iterrows():
                if track_idx > 0:
                    prev_row = track_df.iloc[track_idx - 1]
                    start = dt.datetime.fromtimestamp(prev_row.gps_epoch, dt.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')
                    end = dt.datetime.fromtimestamp(row.gps_epoch, dt.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')
                    track_data[track_idx] = dict(
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
        surface_events_df = surface_events_df.replace({np.nan: None})

        # calculate previous 24 hours
        t24h = pd.to_datetime(ts_now) - pd.Timedelta(hours=24)

        surface_events_dict = dict()
        currents_dict = dict()
        total_iridium_seconds = 0

        # build the information for the surfacings and depth-averaged currents
        for idx, row in surface_events_df.iterrows():
            se = row.to_dict()

            total_iridium_seconds = total_iridium_seconds + se['call_length_seconds']
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

            vx_test = isinstance(surfacing_m_water_vx, float)
            vy_test = isinstance(surfacing_m_water_vy, float)
            mag_test = isinstance(surfacing_m_gps_mag_var, float)
            if np.logical_and(np.logical_and(vx_test, vy_test), mag_test):
                # units: current_bearing = degrees, current_speed = m/s
                current_bearing, current_speed = uv2spdir(surfacing_m_water_vx, surfacing_m_water_vy,
                                                          mag=-surfacing_m_gps_mag_var)
            elif np.logical_and(vx_test, vy_test):
                # calculate m_gps_mag_var if it's not there
                calculator = MagneticFieldCalculator(model='igrf')
                result = calculator.calculate(latitude=se['gps_lat_degrees'],
                                              longitude=se['gps_lon_degrees'],
                                              altitude=0,
                                              date=currents_folder_name)
                mag = -result['field-value']['declination']['value']
                current_bearing, current_speed = uv2spdir(surfacing_m_water_vx, surfacing_m_water_vy, mag=-mag)
            else:
                current_bearing = None
                current_speed = None

            # calculate where the glider will be in 1 day floating at surface
            try:
                distance_1day_km = current_speed * 86400 / 1000
                dest_1day = geopy.distance.distance(kilometers=distance_1day_km).destination((se['gps_lat_degrees'], se['gps_lon_degrees']), bearing=current_bearing)
                currents_dict[currents_folder_name][idx]['lon_degrees_end'] = dest_1day.longitude
                currents_dict[currents_folder_name][idx]['lat_degrees_end'] = dest_1day.latitude
            except TypeError:
                # if end lat/lon can't be calculated, remove this record from the currents dictionary
                del currents_dict[currents_folder_name][idx]

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

            if idx == 0:  # deployment location
                surface_events_dict[folder_name][idx]['surface_event_popup']['current_speed'] = format_float(current_speed)  # m/s
                surface_events_dict[folder_name][idx]['surface_event_popup']['current_speed_bearing'] = format_int(current_bearing)

                deployment_popup_dict = surface_events_dict[folder_name][idx]['surface_event_popup']
                deployment_ts_Z = dt.datetime.fromtimestamp(se['connect_time_epoch'], dt.UTC).strftime(
                    '%Y-%m-%dT%H:%M:%SZ')
                deployment_gps_lat_degrees = se['gps_lat_degrees']
                deployment_gps_lon_degrees = se['gps_lon_degrees']

            else:
                # calculate dive/speed from previous surfacing info
                prev_row = surface_events_df.iloc[idx - 1].to_dict()

                # calculate time elapsed and distance since last surfacing
                time_elapsed_seconds = se['connect_time_epoch'] - prev_row['disconnect_time_epoch']
                time_elapsed_minutes = format_float(time_elapsed_seconds / 60)

                distance_travelled = format_float(geopy.distance.geodesic((prev_row['gps_lat_degrees'], prev_row['gps_lon_degrees']),
                                                                          (se['gps_lat_degrees'], se['gps_lon_degrees'])).km)

                # calculate total speed and bearing
                total_speed = distance_travelled * 1000 / time_elapsed_seconds

                total_bearing = calculate_compass_bearing((prev_row['gps_lat_degrees'], prev_row['gps_lon_degrees']),
                                                          (se['gps_lat_degrees'], se['gps_lon_degrees']))

                # calculate glide speed and bearing
                try:
                    cu, cv = spdir2uv(current_speed, current_bearing, deg=True)
                    tu, tv = spdir2uv(total_speed, total_bearing, deg=True)
                    glide_bearing, glide_speed = uv2spdir(tu - cu, tv - cv)
                except TypeError:
                    glide_bearing = None
                    glide_speed = None

                # add dive and current information to the surfacing event (time, distance, speed)
                surface_events_dict[folder_name][idx]['surface_event_popup']['time_elapsed'] = time_elapsed_minutes  # minutes
                surface_events_dict[folder_name][idx]['surface_event_popup']['dist'] = distance_travelled  # km
                surface_events_dict[folder_name][idx]['surface_event_popup']['total_speed'] = format_float(total_speed)  # m/s
                surface_events_dict[folder_name][idx]['surface_event_popup']['total_speed_bearing'] = format_int(total_bearing)
                surface_events_dict[folder_name][idx]['surface_event_popup']['current_speed'] = format_float(current_speed)  # m/s
                surface_events_dict[folder_name][idx]['surface_event_popup']['current_speed_bearing'] = format_int(current_bearing)
                surface_events_dict[folder_name][idx]['surface_event_popup']['glide_speed'] = format_float(glide_speed)  # m/s
                surface_events_dict[folder_name][idx]['surface_event_popup']['glide_speed_bearing'] = format_int(glide_bearing)

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
            cwpt_lat_degrees=format_coordinates(ls_api['waypoint_lat'], decimal_degrees=True),
            cwpt_lon_degrees=format_coordinates(ls_api['waypoint_lon'], decimal_degrees=True),
            distance_flown_km=distance_flown_km,
            days_deployed=days_deployed,
            iridium_mins=format_int(np.round(total_iridium_seconds / 60)),
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

    kmz_filename = convert_kml_to_kmz(savefile, kmz_file_path=None)
    print(f'kmz file location: {kmz_filename}')

    # remove the kml file if the kmz file was written
    if os.path.isfile(kmz_filename):
        os.remove(savefile)


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
                                 'vectors of depth-averaged currents, and "deployed_uv_ts": same as deployed_ts with'
                                 'vectors of depth-averaged currents.',
                            choices=['deployed', 'deployed_ts', 'deployed_uv', 'deployed_uv_ts'],
                            default='deployed')

    arg_parser.add_argument('-s', '--savedir',
                            type=str,
                            help='Save directory',
                            default='/www/web/rucool/gliders/kmz')  # https://rucool.marine.rutgers.edu/gliders/kmz/

    arg_parser.add_argument('-l', '--loglevel',
                            help='Verbosity level',
                            type=str,
                            choices=['debug', 'info', 'warning', 'error', 'critical'],
                            default='info')

    parsed_args = arg_parser.parse_args()

    sys.exit(main(parsed_args))
