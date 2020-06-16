#File for building the dataset needed for the network

import pandas as pd
import os
from functools import reduce

lString = "Langeland -"

files = []
name = []
times = set([])
times = False


for file in os.listdir('.'):
    filename = os.fsdecode(file)
    if (filename.endswith('.csv')):
        data = pd.read_csv(filename, index_col=None, names=['timestamp', 'val'])


        # if (not ts):
        #     files.append(data['timestamp'])
        #     name.append('Timestamps')
        #     ts = True
        
        # times.add(data['timestamp'])

        
        files.append(data)
        if filename.startswith(lString):
            filename = filename[len(lString):]

        filename = filename[:-4]
        name.append(filename)

print(name)


# https://stackoverflow.com/questions/23668427/pandas-three-way-joining-multiple-dataframes-on-columns
df = reduce(lambda left,right: pd.merge(left,right,on='timestamp'), files)

df = df.drop('timestamp', axis=1)

df.columns = name

df.to_csv('../../Models/data.csv', index=None, header=name) 

print(df.tail())


