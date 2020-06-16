import pandas as pd
import geopy.distance
import math

from functools import reduce

def convertToDD(dms):
    
    dmsString = str(dms)

    # print('Converting: '+dmsString)

    degrees = int(dmsString[:2])
    minutes = float(dmsString[2:8])

    dd = degrees + (minutes/60)
    return dd

#Should be done nicer and automatically in the future

fuel1 = pd.read_csv('../Langeland - Fuel DG 1 lh.csv', names=['timestamp', 'fuel1'])

fuel2 = pd.read_csv('../Langeland - Fuel DG 2 lh.csv', names=['timestamp', 'fuel2'])

fuel3 = pd.read_csv('../Langeland - Fuel DG 3 lh.csv', names=['timestamp', 'fuel3'])

fuel4 = pd.read_csv('../Langeland - Fuel DG 4  lh.csv', names=['timestamp', 'fuel4'])

fuel5 = pd.read_csv('../Langeland - Fuel DG 5 lh.csv', names=['timestamp', 'fuel5'])

longitude = pd.read_csv('../Langeland - GPS Longitude.csv', names=['timestamp', 'lon'])

latitude = pd.read_csv('../Langeland - GPS Latitude.csv', names=['timestamp', 'lat'])

speed = pd.read_csv('speeds.csv', names=['timestamp', 'speed'])


dfs = [fuel1, fuel2, fuel3, fuel4, fuel5, longitude, latitude, speed]
df_final = reduce(lambda left,right: pd.merge(left,right,on='timestamp'), dfs)



last_fuel = -1
last_lo = -1
last_la = -1


# fuel = []

total_fuel = 0
total_dist = 0


comb = []


for index, row in df_final.iterrows():
    cFuel = 0

    ts = row['timestamp']

    # Sum up all the fuel that will be used 
    f1 = row['fuel1']
    if (f1 >= 0.0):
        cFuel = cFuel + f1

    f2 = row['fuel2']
    if (f2 >= 0.0):
        cFuel = cFuel + f2

    f3 = row['fuel3']
    if (f3 >= 0.0):
        cFuel = cFuel + f3
        
    f4 = row['fuel4']
    if (f4 >= 0.0):
        cFuel = cFuel + f4

    f5 = row['fuel5']
    if (f5 >= 0.0):
        cFuel = cFuel + f5

    cur_la = row['lat']
    cur_lo = row['lon']

    # Convert from knots to km/h 
    cur_speed = (row['speed']) * 1.852


    cur_la = convertToDD(cur_la)
    cur_lo = convertToDD(cur_lo)

    if (last_la != -1 and last_lo != -1):

        cord1 = (last_la, last_lo)
        cord2 = (cur_la, cur_lo)

        

        uFuel = math.floor((cFuel+last_fuel)/2)


        dist = geopy.distance.vincenty(cord1, cord2).km

        # Exlcude outliers/errors 
        if (dist < 1.0):


            # Convert from hour to seconds 
            # Maybe not neccesary since we convert back to hour immediately?
            time = (dist / abs(cur_speed)) * 3600
            
            fuel_in_liter = uFuel * time * (1/3600)

            if (fuel_in_liter < 3):

                data = [ts, fuel_in_liter]
                comb.append(data)
            
    last_la = cur_la
    last_lo = cur_lo
    last_fuel = cFuel



df = pd.DataFrame(comb, columns=['timestamp', 'fuel_consumption'])
df.to_csv("fuel_consumption.csv", index=None, header=False)

print(df.tail())



