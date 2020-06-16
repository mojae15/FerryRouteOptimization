import pandas as pd
import numpy as np
import math
from functools import reduce

relative_dir = pd.read_csv('../Langeland -Wind Direction Relative °.csv', names=['timestamp', 'relative_dir'])
heading = pd.read_csv('../Langeland - Heading True °.csv', names=['timestamp', 'heading'])

fSpeed = pd.read_csv('../Langeland - Longitudinal Ground Speed Knots.csv', names=['timestamp', 'fSpeed'])

wSpeed = pd.read_csv('../Langeland - Wind Speed Relative Knots.csv', names=['timestamp', 'wSpeed'])

dfs = [relative_dir, heading, fSpeed, wSpeed]
df_final = reduce(lambda left,right: pd.merge(left,right,on='timestamp'), dfs)


actual_dir = []
actual_speed = []

for index, row in df_final.iterrows():
    cHeading = row['heading']

    cFSpeed = row['fSpeed']

    cWDir = row['relative_dir']
    
    cWSpeed = row['wSpeed']


    #Create relative wind vector
    wX = cWSpeed * math.cos(cWDir)

    wY = cWSpeed * math.sin(cWDir)
    windV = np.array([wX, wY])

    #Create ferry vector
    fX = cFSpeed * math.cos(cHeading)

    fY = cFSpeed * math.sin(cHeading)
    ferryV = np.array([fX, fY])

    #Get actual wind vector
    actualV = np.add(windV, ferryV)

    aDir = (cWDir + cHeading) % 360


    #Magnitude of this vector should be the actual wind speed

    aSpeed = np.linalg.norm(actualV)


    aDir = (cWDir + cHeading) % 360

    dirData = [row['timestamp'], aDir]
    speedData = [row['timestamp'], aSpeed]

    actual_dir.append(dirData)
    actual_speed.append(speedData)
    

df = pd.DataFrame(actual_dir, columns=['timestamp', 'Wind Direction'])
df.to_csv("wind_direction.csv", index=None, header=False)


df = pd.DataFrame(actual_speed, columns=['timestamp', 'Wind Speed'])
df.to_csv("wind_speed.csv", index=None, header=False)

