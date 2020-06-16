import pandas as pd
import folium
from functools import reduce
from geopy import distance

def avg(lst):
    return sum(lst) / len(lst)

def convertToDD(dms):
    
    dmsString = str(dms)

    # print('Converting: '+dmsString)

    degrees = int(dmsString[:2])
    minutes = float(dmsString[2:8])

    dd = degrees + (minutes/60)
    return dd



lat = pd.read_csv('../Langeland - GPS Latitude.csv', names=['timestamp', 'lat'])

lon = pd.read_csv('../Langeland - GPS Longitude.csv', names=['timestamp', 'lon'])

speed = pd.read_csv('../Langeland - Longitudinal Water Speed Knots.csv', names=['timestamp', 'speed'])

fuel1 = pd.read_csv('../Langeland - Fuel DG 1 lh.csv', names=['timestamp', 'fuel1'])

fuel2 = pd.read_csv('../Langeland - Fuel DG 2 lh.csv', names=['timestamp', 'fuel2'])

fuel3 = pd.read_csv('../Langeland - Fuel DG 3 lh.csv', names=['timestamp', 'fuel3'])

fuel4 = pd.read_csv('../Langeland - Fuel DG 4  lh.csv', names=['timestamp', 'fuel4'])

fuel5 = pd.read_csv('../Langeland - Fuel DG 5 lh.csv', names=['timestamp', 'fuel5'])

depth = pd.read_csv('../Langeland - DPT - Depth m.csv', names=['timestamp', 'depth'])

dfs = [lat, lon, speed, fuel1, fuel2, fuel3, fuel4, fuel5, depth]

df_final = reduce(lambda left,right: pd.merge(left,right,on='timestamp'), dfs)


m = folium.Map(location=[54.55, 10.50])

begun_route = False
done = False

last_lat = -1
last_lon = -1

distances = []
speeds = []
depths = []

starttime = 0
endtime = 0

fuel_used = 0

fuel_in_liter = 0

total_fuel = 0

for index, row in df_final.iterrows():
    if (not done):
        if (not begun_route):
            # Started a route
            if (row['speed'] > 10.0):
                cur_la = convertToDD(row['lat'])
                cur_lo = convertToDD(row['lon'])
                folium.Marker([cur_la, cur_lo]).add_to(m)
                last_lat = cur_la
                last_lon = cur_lo

                print("Plotting point: "+str(cur_la)+", "+str(cur_lo))

                starttime = pd.Timestamp(row['timestamp'])
                speeds.append(row['speed'])
                depths.append(row['depth'])
                
                begun_route = True
        else:
            if (row['speed'] > 10.0):
                cur_la = convertToDD(row['lat'])
                cur_lo = convertToDD(row['lon'])
                folium.Marker([cur_la, cur_lo]).add_to(m)

                p1 = (cur_la, cur_lo)
                p2 = (last_lat, last_lon)
                dist = distance.distance(p1, p2).km * 1000
                distances.append(dist)
                
                last_lat = cur_la
                last_lon = cur_lo
                speeds.append(row['speed'])
                depths.append(row['depth'])


                cFuel = 0
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

                
                dist = distance.distance(p1, p2).km
                if (dist < 1.0):
                    # Convert from knots to km/h 
                    cur_speed = (row['speed']) * 1.852

                    time = (dist / abs(cur_speed)) * 3600
            
                    fuel_in_liter =  cFuel * time * (1/3600)

                    print("Fuel used on this point: "+str(fuel_in_liter))

                    total_fuel = total_fuel + fuel_in_liter





                print("Plotting point: "+str(cur_la)+", "+str(cur_lo))

            else:
                endtime = pd.Timestamp(row['timestamp'])
                done = True        
    


# print("Average: " + str(avg(distances)))
print("Average: "+str(avg(depths)))

# print("Average: " + str(avg(speeds)))
# transittime = (endtime - starttime)
# print(transittime)


# print(total_fuel)

m.save('path.html')




