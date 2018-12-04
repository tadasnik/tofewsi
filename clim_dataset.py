import os
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import dask as ds
from dask.distributed import Client

class Climdata(object):

    def __init__(self, data_path, bbox=None, hour=None):
        self.data_path = data_path
        self.bbox = bbox
        self.hour = hour

    def read_dataset(self, file_name):
        """
        Reads netCDF dataset using xarray.
        Args:
            parameter - (int) grib_id of the dataset to read.
        Returns
            xarray dataframe
        """
        dataset_path = os.path.join(self.data_path, file_name)
        dataset = xr.open_dataset(dataset_path)
        if self.bbox:
            dataset = self.spatial_subset(dataset, self.bbox)
        if self.hour:
            dataset = self.time_subset(dataset, self.hour)
        return dataset

    def spatial_subset(self, dataset, bbox):
        """
        Selects data within spatial bbox. bbox coords must be given as
        positive values for the Northern hemisphere, and negative for
        Southern. West and East both positive - Note - the method is
        naive and will only work for bboxes fully fitting in the Eastern hemisphere!!!
        Args:
            dataset - xarray dataset
            bbox - (list) [North, South, West, East]
        Returns:
            xarray dataset
        """
        dataset = dataset.sel(longitude = slice(bbox[2], bbox[3]))
        dataset = dataset.sel(latitude = slice(bbox[0], bbox[1]))
        return dataset

    def time_subset(self, dataset, hour=None, start_date=None, end_date=None):
        """
        Selects data within spatial bbox.
        Args:
            dataset - xarray dataset
            hour - (int) hour
        Returns:
            xarray dataset
        """
        if hour:
            dataset = dataset.sel(time=datetime.time(hour))
        return dataset

    def wind_speed(self, dataset):
        dataset['w10'] = np.sqrt(dataset['u10']**2 + dataset['v10']**2)
        return dataset

    def relative_humidity(self, dataset):
        """
        Relative humidity is calculated from 2 metre temperature and
        2 metre dewpoint temperature using the August-Roche-Magnus approximation
        RH = 100 * (EXP((17.625*TD)/(243.04+TD)) / EXP((17.625*T)/(243.04+T)))
        where TD is dewpoint temperature and T is temperature
        """
        celcius_d2m = dataset['d2m'] - 273.15
        celcius_t2m = dataset['t2m'] - 273.15
        top = np.exp((17.625 * celcius_d2m) / (243.04 + celcius_d2m))
        bot = np.exp((17.625 * celcius_t2m) / (243.04 + celcius_t2m))
        dataset['h2m'] = 100 * (top / bot)
        return dataset

    def prepare_xarray_fwi(self, an_fname, fc_fname):
        an_dataset = self.read_dataset(an_fname)
        an_dataset = self.time_subset(an_dataset, hour = 7)
        an_dataset = self.wind_speed(an_dataset)
        an_dataset = self.relative_humidity(an_dataset)

        fc_dataset = self.read_dataset(fc_fname)
        preci = fc_dataset['tp'].resample(time = '24H',
                                          closed = 'right',
                                          label = 'right',
                                          base = 7).sum(dim = 'time')
        # converting total precipitation to mm from m
        preci = preci * 1000

        #converting K to C
        an_dataset['t2m'] = an_dataset['t2m'] - 273.15
        fwi_darray = xr.merge([an_dataset[['t2m', 'w10', 'h2m']], preci[:-1]])
        return fwi_darray

    def era5_dataset(self, an_fname, fc_fname):
        """
        Read era5 analysis (an_fname) and forecast (fc_fname)
        datasets, derive wind_speed, rel_hum, merge and return
        dataset with ['t2m', 'w10', 'h2m', 'ssrd' and 'tp']
        variables
        """
        an_dataset = self.read_dataset(an_fname)
        an_dataset = self.wind_speed(an_dataset)
        an_dataset = self.relative_humidity(an_dataset)
        fc_dataset = self.read_dataset(fc_fname)
        dataset = xr.merge([an_dataset[['t2m', 'w10', 'h2m']],
                            fc_dataset[['ssrd', 'tp']]])
        return dataset

    def prepare_dataframe_era5_chirps(self, dataset):
        dfr = dataset.to_dataframe()
        dfr.reset_index(inplace=True)
        dfr.loc[:, 'Day'] = dfr['time'].dt.day
        dfr.loc[:, 'Hour'] = dfr['time'].dt.hour
        dfr.loc[:, 'Month'] = dfr['time'].dt.month
        dfr.loc[:, 'Year'] = dfr['time'].dt.year
        dfr.drop('time', axis=1, inplace=True)
        dfr = dfr[['latitude', 'longitude', 'Day', 'Hour',
                             'Month', 'Year', 'ssrd', 't2m', 'w10', 'h2m', 'tp']]
        dfr['ssrd'] = dfr['ssrd'].astype(int)
        cols = ['lat', 'long', 'Day', 'Hour', 'Month', 'Year',
                'Incident Shortwave Radiation', 'Air Temperature',
                'Windspeed', 'Humidity', 'Precipitation']
        dfr.columns = cols
        dfr = dfr.round({'lat': 3,
                         'long': 3,
                         'Air Temperature': 1,
                         'Windspeed': 1,
                         'Humidity': 1,
                         'Precipitation': 1})
        return dfr



    def prepare_dataframe_era5(self, an_fname, fc_fname):
        an_dataset = self.read_dataset(an_fname)
        an_dataset = self.wind_speed(an_dataset)
        an_dataset = self.relative_humidity(an_dataset)
        an_dfr = an_dataset[['t2m', 'w10', 'h2m']].to_dataframe()
        an_dfr.reset_index(inplace=True)

        fc_dataset = self.read_dataset(fc_fname)
        fc_dfr = fc_dataset[['ssrd', 'tp']].to_dataframe()
        fc_dfr.reset_index(inplace=True)

        combined = an_dfr.merge(fc_dfr)
        combined.loc[:, 'Day'] = combined['time'].dt.day
        combined.loc[:, 'Hour'] = combined['time'].dt.hour
        combined.loc[:, 'Month'] = combined['time'].dt.month
        combined.loc[:, 'Year'] = combined['time'].dt.year
        combined.drop('time', axis=1, inplace=True)

        combined = combined[['latitude', 'longitude', 'Day', 'Hour',
                             'Month', 'Year', 'ssrd', 't2m', 'w10', 'h2m', 'tp']]
        # converting total precipitation to mm from m
        combined.loc[:, 'tp'] = combined['tp'] * 1000
        cols = ['lat', 'long', 'Day', 'Hour', 'Month', 'Year',
                'Incident Shortwave Radiation', 'Air Temperature',
                'Windspeed', 'Humidity', 'Precipitation']
        combined.columns = cols
        return combined


    def write_csv(self, dfr, fname):
        print('writing dataframe to csv file {0}'.format(fname))
        dfr.to_csv(fname, index=False, compression = 'gzip')
        print('finished writing')


if __name__ == '__main__':
    #data_path = '/home/tadas/tofewsi/data/'
    #fname = '2013-12-31_to_2014-12-31_169.128_228.128_0.25deg.nc'
    data_path = '/mnt/data/SEAS5'
    #data_path = '/mnt/data/era5/amazon'
    #data_path = '/mnt/data/era5/riau'
    #chirps_path = '/mnt/data/chirps'
    #data_path = '.'
    #fname = '23_tt_6hourly.nc'
    #fname1 = '24_tt_6hourly.nc'
    ds_name = os.path.join(data_path, '2018_11_seas5.nc')
    # indonesia bbox
    #bbox = [8.0, 93.0, -13.0, 143.0]
    bbox = [8.0, -13.0, 93.0, 143.0]
    cl = Climdata(data_path, bbox=bbox, hour=None)
    """
    # Riau bbox
    bbox = [3, -2, 99, 104]
    #bbox = [1, -.4, 101, 103.5]
    cl = Climdata(data_path, bbox=None, hour=None)
    #dcs = []
    #fwis = []
    year = 2008
    ds = xr.open_dataset(os.path.join('/mnt/data/era5/indonesia', 'era5_ecosys_{0}.nc'.format(year)))
    ds = cl.spatial_subset(ds, bbox)
    ch = xr.open_dataset(os.path.join(chirps_path, 'chirps-v2.0.{0}.days_p05.nc'.format(year)))
    ch = cl.spatial_subset(ch, [bbox[1], bbox[0], bbox[2], bbox[3]])
    """
    """
    for year in range(2009, 2016, 1):
        print(year)
        ds = xr.open_dataset(os.path.join('/mnt/data/era5/indonesia', 'era5_ecosys_{0}.nc'.format(year)))
        ds = cl.spatial_subset(ds, bbox)
        ds['ssrd'][:7, :, :] = ds['ssrd'][24:31, :, :].values
        ds['tp'][:7, :, :] = ds['tp'][24:31, :, :].values
        ds = ds.sel(time=ds['time.year'] == year)
        ch = xr.open_dataset(os.path.join(chirps_path, 'chirps-v2.0.{0}.days_p05.nc'.format(year)))
        ch = cl.spatial_subset(ch, [bbox[1], bbox[0], bbox[2], bbox[3]])
        ch = ch.sortby('latitude', ascending=False)
        ch /= 24.
        dt = ch['time'][-1] + pd.Timedelta('23H')
        darr = xr.DataArray(ch.isel(time=[-1])['precip'].values,
                            coords=[[dt.values], ch.latitude.values, ch.longitude.values],
                            dims=['time', 'latitude', 'longitude'])
        da = darr.to_dataset(name = 'precip')
        ch = xr.concat([ch, da], dim = 'time')
        hc = ch.resample(time="1H", closed='left').ffill()
        #hc = hc.where(hc>=0)
        ds = ds[['t2m', 'w10', 'h2m', 'ssrd']]
        ds = ds.interp_like(hc)
        ds = ds.assign({'tp': hc['precip']})
        ds.to_netcdf(os.path.join('/mnt/data/era5/riau', 'era5_chirps_5km_{0}.nc'.format(year)))
        dfr = cl.prepare_dataframe_era5_chirps(ds)
        cl.write_csv(dfr, os.path.join('/mnt/data/era5/riau', 'era5_chirps_5km_{0}.csv.gz'.format(year)))

    """
    """
        #fwi_ds = xr.open_dataset('data/fwi_dc_riau_{0}.nc'.format(year))
        fwi_ds = ds.read_dataset('data/fwi_dc_riau_{0}.nc'.format(year))
        #fwi = fwi_ds['fwi'].where((fwi_ds['latitude']==1.25)&(fwi_ds['longitude']==101.5), drop=True)
        #dc = fwi_ds['dc'].where((fwi_ds['latitude']==1.25)&(fwi_ds['longitude']==101.5), drop=True)
        dcs.append(fwi_ds['dc'])
        fwis.append(fwi_ds['fwi'])
        fwi = xr.concat(fwis, dim = 'time')
        dc = xr.concat(dcs, dim = 'time')
        print(year)
        an_fname = '{0}-01-01_{0}-12-31_165.128_166.128_167.128_168.128_0.25deg.nc'.format(year)
        fc_fname = '{0}-01-01_{0}-12-31_169.128_228.128_0.25deg.nc'.format(year)
        ds = cl.era5_dataset(an_fname, fc_fname)
        ds.to_netcdf(os.path.join(data_path, 'era5_ecosys_{0}.nc'.format(year)))
        """
        #fwi_darray = ds.prepare_xarray_fwi(an_fname, fc_fname)
        #fwi_name = 'rh_temp_wind_prcp_amazon_{0}.nc'.format(year)
        #fwi_darray.to_netcdf(os.path.join('data', fwi_name))
        #dfr = ds.prepare_dataframe_era5(an_fname, fc_fname)
        #dfr.to_pickle('/home/tadas/tofewsi/data/era5_ecosys_{0}'.format(year))
        #ds.write_csv(dfr, 'era5_{0}_riau.csv'.format(year))
