import pandas as pd
from geopy import distance


speed = pd.read_csv('../Langeland - Longitudinal Water Speed Knots.csv', names=['timestamp', 'speed'])

df = speed[speed['speed'] > 10.0]

df.to_csv('speeds.csv', index=None, header=False)

print(df.head())