import pandas as pd
from functools import reduce
import os


dataList = []


for file in os.listdir('.'):
    filename = os.fsdecode(file)
    if (filename.endswith('.csv')):
        df = pd.read_csv(filename, header=0)

        for index, row in df.iterrows():
            curMag = row['Max. current magnitude [knots]']
            curDir = row['Current direction dominant [degrees]']

            #Probably a better way to do this
            if (curDir == 'North'):
                curDir = 0.0
            elif (curDir == 'East'):
                curDir = 90.0
            elif (curDir == 'South'):
                curDir = 180.0
            elif (curDir == 'West'):
                curDir = 270.0
                

            ts = row['Start time']

            data = (ts, curMag, curDir)
            dataList.append(data)


final_df = pd.DataFrame(dataList, columns=['timestamp', 'current_magnitude', 'current_direction'])
final_df = final_df.sort_values(by='timestamp')

final_df['timestamp'] = pd.to_datetime(final_df['timestamp'])
final_df.index = final_df['timestamp']
del final_df['timestamp']

df_pad = final_df.resample('15S').pad()
df_pad['timestamp'] = df_pad.index

cols = list(df_pad)

cols = cols[-1:] + cols[:-1]
df_pad = df_pad[cols]


print(df_pad)
mag_df = df_pad.filter(['timestamp', 'current_magnitude'], axis=1)

dir_df = df_pad.filter(['timestamp', 'current_direction'], axis=1)



mag_df.to_csv('../Data/Relevant_Data/current_magnitude.csv', index=None, header=False)

dir_df.to_csv('../Data/Relevant_Data/current_direction.csv', index=None, header=False)


