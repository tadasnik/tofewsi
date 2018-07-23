import pandas as pd

#Coordinates for the sites (lon, lat)
coords = {'acacia': [101.49, 1.3], 
          'forest': [101.4, 1.27],
          'rubber': [101.44, 1.39]}

fname = '/home/tadas/tofewsi/docs/groundwater_Riau.xlsx'
gr = pd.read_excel(fname)
gr['Date'] = gr['Date'].astype(str)
gr['Time'] = gr['Time'].astype(str)
gtimes = pd.to_datetime(gr['Date'] + gr['Time'], format="%d.%m.%Y%H:%M:%S")

gr.loc[:, 'Day'] = gtimes.dt.day
gr.loc[:, 'Hour'] = gtimes.dt.hour
gr.loc[:, 'Month'] = gtimes.dt.month
gr.loc[:, 'Year'] = gtimes.dt.year
gr.drop('Time', axis=1, inplace=True)
gr.drop('Date', axis=1, inplace=True)


df9 = pd.read_pickle('data/era5_ecosys_2009')
df10 = pd.read_pickle('data/era5_ecosys_2010')
dfr = pd.concat((df9, df10))
dfr = dfr.loc[dfr['long'] == 101.5]
adfr = dfr.loc[dfr['lat'] == 1.25]
rdfr = dfr.loc[dfr['lat'] == 1.5]

acacia = gr[gr['Acacia'] != 999.0][['Acacia', 'Day', 'Hour', 'Month', 'Year']]
forest = gr[gr['Forest'] != 999.0][['Forest', 'Day', 'Hour', 'Month', 'Year']]
rubber = gr[gr['Rubber'] != 999.0][['Rubber', 'Day', 'Hour', 'Month', 'Year']]

acacia = pd.merge(adfr, acacia, on = ['Year', 'Month', 'Day', 'Hour'])
forest = pd.merge(adfr, forest, on = ['Year', 'Month', 'Day', 'Hour'])
rubber = pd.merge(rdfr, rubber, on = ['Year', 'Month', 'Day', 'Hour'])

acacia.rename(columns={'Acacia': 'Ground_water'}, inplace=True)
forest.rename(columns={'Forest': 'Ground_water'}, inplace=True)
rubber.rename(columns={'Rubber': 'Ground_water'}, inplace=True)

acacia.to_pickle('data/acacia_era5_ground_water')
forest.to_pickle('data/forest_era5_ground_water')
rubber.to_pickle('data/rubber_era5_ground_water')

