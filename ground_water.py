import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from clim_dataset import *

#Coordinates for the sites (lon, lat)
"""
coords = {'acacia': [101.49, 1.3],
          'forest': [101.4, 1.27],
          'rubber': [101.44, 1.39]}
"""
fba = [0.68, 102.262]
cde = [0.875, 102.34]

data_path = '/mnt/data/era5/glob'
bbox = [1, 0, 101.5, 103]
cl = Climdata(data_path, bbox=bbox, hour=None)
fbadfs = []
cdedfs = []
for year in range(2008, 2019, 1):
    print(year)
    for month in range(1, 13, 1):
        print(month)
        dts = xr.open_dataset('/mnt/data/era5/glob/{0}_{1}.nc'.format(year, month))
        dts = cl.spatial_subset(dts, bbox)
        #dts.to_netcdf('/home/tadas/tofewsi/data/ground/{0}_{1}.nc'.format(year, month))
        dts = dts.load()
        dts = cl.wind_speed(dts)
        dts = cl.relative_humidity(dts)
        datas = dts[['ssrd', 't2m', 'w10', 'h2m', 'tp']].interp(latitude = fba[0], longitude = fba[1])
        datas = datas.to_dataframe()
        datas = datas.reset_index()
        datas.loc[:, 'Day'] = datas.time.dt.day
        datas.loc[:, 'Month'] = datas.time.dt.month
        datas.loc[:, 'Year'] = datas.time.dt.year
        datas.loc[:, 'Hour'] = datas.time.dt.hour
        datas.drop('time', axis=1, inplace=True)
        datas = datas[['latitude', 'longitude', 'Day', 'Hour',
                             'Month', 'Year', 'ssrd', 't2m', 'w10', 'h2m', 'tp']]
        # converting total precipitation to mm from m
        datas.loc[:, 'tp'] = datas['tp'] * 1000
        cols = ['lat', 'long', 'Day', 'Hour', 'Month', 'Year',
                'Incident Shortwave Radiation', 'Air Temperature',
                'Windspeed', 'Humidity', 'Precipitation']
        datas.columns = cols
        fbadfs.append(datas)

        datas = dts[['ssrd', 't2m', 'w10', 'h2m', 'tp']].interp(latitude = cde[0], longitude = cde[1])
        datas = datas.to_dataframe()
        datas = datas.reset_index()
        datas.loc[:, 'Day'] = datas.time.dt.day
        datas.loc[:, 'Month'] = datas.time.dt.month
        datas.loc[:, 'Year'] = datas.time.dt.year
        datas.loc[:, 'Hour'] = datas.time.dt.hour
        datas.drop('time', axis=1, inplace=True)
        datas = datas[['latitude', 'longitude', 'Day', 'Hour',
                             'Month', 'Year', 'ssrd', 't2m', 'w10', 'h2m', 'tp']]
        # converting total precipitation to mm from m
        datas.loc[:, 'tp'] = datas['tp'] * 1000
        cols = ['lat', 'long', 'Day', 'Hour', 'Month', 'Year',
                'Incident Shortwave Radiation', 'Air Temperature',
                'Windspeed', 'Humidity', 'Precipitation']
        datas.columns = cols
        cdedfs.append(datas)
fbadfr = pd.concat(fbadfs)
cl.write_csv(fbadfr, '/home/tadas/tofewsi/data/fba_sites_era5_ecosys.csv'.format(year))
cdedfr = pd.concat(fbadfs)
cl.write_csv(cdedfr, '/home/tadas/tofewsi/data/cde_sites_era5_ecosys.csv'.format(year))










"""


    fwi_ds = xr.open_dataset('data/fwi_dc_riau_{0}.nc'.format(year))
    fwi = fwi_ds['fwi'].where((fwi_ds['latitude']==1.25)&(fwi_ds['longitude']==101.5), drop=True)
    dc = fwi_ds['dc'].where((fwi_ds['latitude']==1.25)&(fwi_ds['longitude']==101.5), drop=True)
    dcs.append(dc)
    fwis.append(fwi)

ddc = xr.concat(dcs, dim='time')
ffwi = xr.concat(fwis, dim='time')

dfr = pd.read_pickle('data/forest_era5_ground_water')
dfr['dtime'] = pd.to_datetime(dfr[['Year', 'Month', 'Day', 'Hour']])

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



for year in [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015]:
    dfr = pd.read_pickle('data/era5_ecosys_{0}'.format(year))
    dfr = dfr.loc[dfr['long'] == 101.5]
    adfr = dfr.loc[dfr['lat'] == 1.25]
    rdfr = dfr.loc[dfr['lat'] == 1.5]
    adfr.to_csv('data/acacia_forest_{0}.csv'.format(year), index=False, float_format='%.2f')
    rdfr.to_csv('data/rubber_{0}.csv'.format(year), index=False, float_format='%.2f')

"""



