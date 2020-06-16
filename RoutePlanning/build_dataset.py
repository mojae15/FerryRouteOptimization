import sys
import pandas as pd
import requests
from time import sleep
import csv
import netCDF4
from netCDF4 import Dataset
import numpy as np
import math
from datetime import datetime



#Download data from openweathermap.org, and save it to a .csv file
#Takes a little while since it has to sleep to prevent the program from being locked out of the api


direction = sys.argv[1]



def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text 

OW_API_KEY = '1a5172ef6705165fe61df624195ef540'


values = []

#Download FCOO Data
url = "https://wms.fcoo.dk/webmap/FCOO/GETM/idk-600m_3D-velocities_surface_1h.DK600-v007C.nc"
r = requests.get(url, allow_redirects=True)
open('idk-600m_3D-velocities_surface_1h.DK600-v007C.nc', 'wb').write(r.content)

rootgrp = Dataset('idk-600m_3D-velocities_surface_1h.DK600-v007C.nc', 'r')

count = 1

dim_data = ()


firstLine = True
with (open('points.csv', newline='')) as csvfile:
    csvreader = csv.reader(csvfile, delimiter= ',')
    for row in csvreader:
        if (firstLine):
            x_dim = row[0]
            y_dim = row[1]
            dim_data = (x_dim, y_dim, 0, 0, 0, 0, 0)
            firstLine = False
        else:

            lat = row[0]
            lon = row[1]
            dist = row[2]

            #Get data from openweathermap.org

            ow_api_call = 'http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}'.format(lat=lat, lon=lon, api_key=OW_API_KEY)

            response = requests.get(ow_api_call);

            ow_data = response.json()

            wind_data = ow_data['wind']

            wind_speed = wind_data['speed']

            wind_degree = wind_data['deg']

            #Read the .nc file for the current data

            nlats = len(rootgrp.dimensions['latc'])

            lon_arr = rootgrp.variables['lonc'][:]

            lat_arr = rootgrp.variables['latc'][:]

            time_arr = rootgrp.variables['time'][:]


            lon_index = np.abs(lon_arr - float(lon)).argmin()

            lat_index = np.abs(lat_arr - float(lat)).argmin()


            text = rootgrp.variables['time'].units

            time_string = remove_prefix(text, 'seconds since ')

            dt = datetime.strptime(time_string, '%Y-%m-%d %H:%M:%S')

            now = datetime.now()

            total_seconds = math.floor((now-dt).total_seconds())

            time_index = np.abs(time_arr - total_seconds).argmin()

            ## x-direction
            uu = rootgrp.variables['uu']

            uu_res = uu[time_index, lat_index, lon_index]


            ##y-direction
            vv = rootgrp.variables['vv']

            vv_res = vv[time_index, lat_index, lon_index]


            total_velocity = math.sqrt((uu_res*uu_res) + (vv_res*vv_res))

            total_angle = math.degrees(math.atan(abs(vv_res) / abs(uu_res)))

            data = (lat, lon, dist, wind_speed, wind_degree, total_velocity, total_angle)


            values.append(data)

            if (count % 500 == 0):
                print("Sleeping")
                print(count)
                sleep(60)

            count = count + 1



if (direction == 2):
    values.reverse()

values.insert(0, dim_data)



df = pd.DataFrame(values, columns=['lat', 'lon', 'dist', 'wind_speed',  'wind_degree', 'current_velocity', 'current_angle'])
df.to_csv("data_points.csv", index=None, header=False)

