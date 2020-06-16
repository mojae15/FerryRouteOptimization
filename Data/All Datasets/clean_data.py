import pandas as pd
import os

for file in os.listdir('.'):
    filename = os.fsdecode(file)
    if (filename.endswith('.csv')):
        data = pd.read_csv(filename, names=['timestamp', 'source', 'val'], skiprows=1)

        points = data['source'].unique()

        for source in points:
            to_write = data[data.source == source]

            # Drop the "source" part in the .csv file
            to_write = to_write[['timestamp', 'val']]


            # Convert the timestamp to workable time
            data_interpol = to_write.set_index(['timestamp'])

            data_interpol.index = pd.to_datetime(data_interpol.index, unit='s')

            # Interpolate missing data
            data_interpol = data_interpol.resample('15S').mean()

            data_interpol['val'] = data_interpol['val'].interpolate()

            
            file_name = source + '.csv'

            # #Should probably choose a better name for the files somehow

            file_name = file_name.translate({ord(i) : None for i in '/[]'})
            file_name = '../' + file_name

            # print(file_name)
            data_interpol.to_csv(file_name, index=True, mode='a', header=False)





# thruster1 = data[data.source == points[0]]

# thruster1.to_csv('thruster1.csv', index=False)