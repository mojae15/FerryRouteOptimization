import pandas as pd
import folium
import utm
import numpy as np
import geopy.distance
import math
import os
from functools import reduce
from fractions import gcd

#Earths radius in meters
R_EARTH = 6378137

def convertToDD(dms):
    
    dmsString = str(dms)

    # print('Converting: '+dmsString)

    degrees = int(dmsString[:2])
    minutes = float(dmsString[2:8])

    dd = degrees + (minutes/60)
    return dd

lString = "Langeland -"

files = []
name = []
times = set([])
times = False




m = folium.Map(location=[54.55, 10.50])


# https://stackoverflow.com/questions/7477003/calculating-new-longitude-latitude-from-old-n-meters


# Spodsbjerg side
start_lat = 54.93
start_lon = 10.85305


# Tårs side
target_lat = 54.88165
target_lon = 10.974616666666666


la_points = []
lo_points = []
dists = []


cord1 = (start_lat, start_lon)
cord2 = (target_lat, target_lon)

# Starting distance in meters
start_dist = geopy.distance.distance(cord1, cord2).km * 1000

# Grain in meters
grain = 100




x_grain = 0
y_grain = 0

x_coef = x_grain * 0.0000089
y_coef = y_grain * 0.0000089


best_dist = start_dist

x_dimension = 0
y_dimension = 0


while (best_dist <= start_dist):
    # print("Start dist: "+str(start_dist))
    # print("Best dist: "+str(best_dist))
    start_dist = best_dist

    new_dist = best_dist

    x_dimension = x_dimension + 1;
    y_dimension = 0
    
    while (new_dist <= best_dist):
        y_dimension = y_dimension +1

        new_lat = start_lat + y_coef
        new_lon = start_lon + x_coef / math.cos(start_lat * 0.018)

        new_cord = (new_lat, new_lon)
        new_dist = geopy.distance.distance(new_cord, cord2).km * 1000

        if (new_dist <= best_dist):
            best_dist = new_dist

            la_points.append(new_lat)
            lo_points.append(new_lon)
            dists.append(new_dist)

            # print("Plotting point: "+str(new_lat)+", "+str(new_lon))

            # lat = new_lat
            # lon = new_lon

            y_grain = y_grain - grain
            y_coef = y_grain * 0.0000089

    
    y_grain = 0
    y_coef = y_grain * 0.0000089
    x_grain = x_grain + grain
    x_coef = x_grain * 0.0000089   

    # Check if next column should be created
    new_lat = start_lat + y_coef
    new_lon = start_lon + x_coef / math.cos(start_lat * 0.018)

    new_cord = (new_lat, new_lon)
    best_dist = geopy.distance.distance(new_cord, cord2).km * 1000





f = open("points.txt", "w+")
f.write(str(x_dimension)+","+str(y_dimension-1)+"\n")


i = 0
for (a, o) in zip(la_points, lo_points):
    cur_la = a

    cur_lo = o

    print("Plotting point: "+str(cur_la)+", "+str(cur_lo))

    f.write(""+str(cur_la)+","+str(cur_lo)+","+str(dists[i])+"\n")

    folium.Marker([cur_la, cur_lo],icon=folium.Icon(color='green'), popup="<b>"+ str(i) + "</b>").add_to(m)
    i = i+1
 

f.close()



m.save('map.html')




