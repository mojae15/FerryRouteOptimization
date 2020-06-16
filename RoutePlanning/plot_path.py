import pandas as pd
import folium

def convertToDD(dms):
    
    dmsString = str(dms)

    # print('Converting: '+dmsString)

    degrees = int(dmsString[:2])
    minutes = float(dmsString[2:8])

    dd = degrees + (minutes/60)
    return dd

path = pd.read_csv('path.csv', index_col=None, names=['lat', 'lon', 'id', 'speed'])


m = folium.Map(location=[54.55, 10.50])


for index, row in path.iterrows():
    lon = row['lon']
    lat = row['lat']
    cur_la = lat

    cur_lo = lon

    folium.Marker([cur_la, cur_lo],icon=folium.Icon(color='red'), popup="<b>" + str(row['id']) + ", " + str(row['speed']) + "</b>").add_to(m)


m.save('path.html')




