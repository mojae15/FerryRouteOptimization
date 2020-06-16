import requests
import sys
import netCDF4
from netCDF4 import Dataset
import numpy as np
from datetime import datetime
import math

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text 

OW_API_KEY = '1a5172ef6705165fe61df624195ef540'


lat = float(sys.argv[1])
lon = float(sys.argv[2])

ow_api_call = 'http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}'.format(lat=lat, lon=lon, api_key=OW_API_KEY)

response = requests.get(ow_api_call);

ow_data = response.json()

wind_data = ow_data['wind']

wind_speed = wind_data['speed']

wind_degree = wind_data['deg']

print(wind_speed)
print(wind_degree)


rootgrp = Dataset('idk-600m_3D-velocities_surface_1h.DK600-v007C.nc', 'r')

nlats = len(rootgrp.dimensions['latc'])

lon_arr = rootgrp.variables['lonc'][:]

lat_arr = rootgrp.variables['latc'][:]

time_arr = rootgrp.variables['time'][:]


lon_index = np.abs(lon_arr - lon).argmin()

lat_index = np.abs(lat_arr - lat).argmin()


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

print(total_velocity)
print(total_angle)

text = rootgrp.variables['time'].units

time_string = remove_prefix(text, 'seconds since ')






