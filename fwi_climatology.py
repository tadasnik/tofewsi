import os
import glob
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import dask as ds
from dask.distributed import Client
from envdata import Envdata
import matplotlib.pyplot as plt
from fwi.fwi_vectorized import FWI

def is_amj(month):
    return month <= 11

def get_precipitation(bbox):
    bbox = [2, 95, -5.935, 119]
    #ds = xr.open_dataset('/mnt/data/era5/indonesia/combined_fwi_vars.nc')
    ds = xr.open_dataset('/mnt/data/fwi/forecast/era5_obs_all_vars.nc')
    ds = ds.sel(time = slice('2002-01-01', '2020-01-01'))#['dc']
    land_mask = 'data/era_land_mask.nc'
    land_mask = xr.open_dataset(land_mask)
    land_mask = land_mask.sel(longitude = slice(bbox[1], bbox[3]))
    land_mask = land_mask.sel(latitude = slice(bbox[0], bbox[2]))
    ds = ds.sel(longitude = slice(bbox[1], bbox[3]))
    ds = ds.sel(latitude = slice(bbox[0], bbox[2]))
    #ds = ds.isel(ds['time.month'] < 9)
    #ds = ds.sel(time=is_amj(ds['time.month']))
    tps = ds.where(land_mask['lsm'][0, :, :].values)
    tpy = tps.groupby('time.year').sum('time')
    tpm = tps.resample(time = 'MS', closed = 'right').median(dim = 'time')
    suma  = tpm.where(tpm != 0).sum(dim = ['longitude', 'latitude'])
    med  = tpy.where(tpy != 0).mean(dim = ['longitude', 'latitude'])
    high  = tpm.where(tpm != 0).quantile(.75, dim = ['longitude', 'latitude'])
    low  = tpm.where(tpm != 0).quantile(.25, dim = ['longitude', 'latitude'])
    suma  = tpy.sum(dim = ['longitude', 'latitude'])
    low  = tpm.where(tpm != 0).quantile(.25, dim = ['longitude', 'latitude'])
    high  = tpm.where(tpm != 0).quantile(.75, dim = ['longitude', 'latitude'])
    return tpm


def calc_fwi(fwi_varr, ffmc0 = None, dmc0 = None, dc0 = None):
    latitudes = np.repeat(np.expand_dims(fwi_varr.latitude, 1),
                          fwi_varr.longitude.shape, axis = 1)
    arr_shape = [fwi_varr.dims[x] for x in ['latitude', 'longitude']]
    lats = fwi_varr.latitude.values
    lons = fwi_varr.longitude.values
    times = fwi_varr.time

    #Arrays with initial conditions
    if ffmc0 is None:
        ffmc0 = np.full(arr_shape, 85.0)
        dmc0 = np.full(arr_shape, 6.0)
        dc0 = np.full(arr_shape, 15.0)

    dcps = []
    fwis = []
    ffmcs = []
    fs = FWI(latitudes)
            #Iterrate over time dimension
    for tt in times:
        fwi_sel = fwi_varr.sel(time = tt)
        rel_hum = fwi_sel['h2m'].values
        rel_hum[rel_hum > 100.0] = 100.0
        fs.set_weather(fwi_sel['t2m'].values,
                       rel_hum,
                       fwi_sel['w10'].values,
                       fwi_sel['tp'].values)

        mth, day = fwi_sel['time.month'].values, fwi_sel['time.day'].values
        #print(fs.temp[32,36], fs.prcp[32,36])
        #print(dc0[32,36])
        ffmca = fs.FFMCcalc(ffmc0)
        dmca = fs.DMCcalc(dmc0, mth)
        dca = fs.DCcalc(dc0, mth)
        dcps.append(dca)
        isia = fs.ISIcalc(ffmca)
        buia = fs.BUIcalc(dmca, dca)
        fwia = fs.FWIcalc(isia, buia)
        #print(ffmca[32,36], dmca[32, 36], dca[32,36], fwia[32,36])
        fwis.append(fwia)
        ffmcs.append(ffmca)
        #print(dca[32,36])
        #print(np.any(np.isnan(dca)))
        ffmc0 = ffmca.copy()
        dmc0 = dmca.copy()
        dc0 = dca.copy()
    #print(dcs[0].shape, [x[32,36] for x in dcs])
    dataset = xr.Dataset({'dc': (['time', 'latitude', 'longitude'], dcps),
                          'ffmc': (['time', 'latitude', 'longitude'], ffmcs),
                          'fwi': (['time', 'latitude', 'longitude'], fwis)},
                          coords={'time': times,
                                  'latitude': lats,
                                  'longitude': lons})
    return dataset, ffmc0, dmc0, dc0, dcps



class Weather(Envdata):
    def __init__(self, data_path, bbox=None, hour=None):
        super().__init__(data_path, bbox=bbox, hour=None)
        self.tp_previous = None

    def dask_client(self):
        self.client = Client(processes=False)

    def read_era5_dask(self, fname):
        dataset = xr.open_dataset(fname)
        dataset = self.spatial_subset(dataset, self.bbox)
        return dataset

    def prepare_fwi_vars(self):
        data_path = '/mnt/data/era5/glob/'
        for nr, year in enumerate(range(1985, 2020, 1)):
            for month in range(1, 13, 1):
                fname = os.path.join(data_path, '{0}_{1}.nc'.format(year, month))
                print(fname)
                fwi_vars = self.prepare_xarray_fwi(fname)
                print('ready, writing....')
                out_fname = os.path.join(self.data_path,
                                                'fwi_vars_{0}_{1}.nc'.format(year, month))
                fwi_vars.to_netcdf(out_fname)


    def prepare_xarray_fwi(self, fname):
        dataset = self.read_era5_dask(fname)
        an_dataset = self.time_subset(dataset[['u10', 'v10', 't2m', 'd2m']], hour = 7)
        wind_speed = self.wind_speed(an_dataset)
        wind_speed.name = 'w10'
        rel_hum = self.relative_humidity(an_dataset)
        rel_hum.name = 'h2m'
        preci = dataset['tp'].resample(time = '24H',
                                       closed = 'right',
                                       label = 'right',
                                       base = 7).sum(dim = 'time')
        # converting total precipitation to mm from m
        preci.values *= 1000
        #converting K to C
        an_dataset['t2m'] = an_dataset['t2m'] - 273.15
        print(self.tp_previous)
        print(preci[0].sum())
        if self.tp_previous is not None:
            preci[0] += self.tp_previous
        print(preci[0].sum())
        fwi_darray = xr.merge([an_dataset['t2m'], wind_speed, rel_hum, preci[:-1]])
        self.tp_previous = preci[-1]
        return fwi_darray

    def calc_fwi(self, fwi_varr, ffmc0 = None, dmc0 = None, dc0 = None):
        latitudes = np.repeat(np.expand_dims(fwi_varr.latitude, 1),
                              fwi_varr.longitude.shape, axis = 1)
        arr_shape = [fwi_varr.dims[x] for x in ['latitude', 'longitude']]
        lats = fwi_varr.latitude.values
        lons = fwi_varr.longitude.values
        times = fwi_varr.time

        #Arrays with initial conditions
        if ffmc0 is None:
            ffmc0 = np.full(arr_shape, 85.0)
            dmc0 = np.full(arr_shape, 6.0)
            dc0 = np.full(arr_shape, 15.0)

        dcps = []
        fwis = []
        ffmcs = []
        fs = FWI(latitudes)
                #Iterrate over time dimension
        for tt in times:
            fwi_sel = fwi_varr.sel(time = tt)
            rel_hum = fwi_sel['h2m'].values
            rel_hum[rel_hum > 100.0] = 100.0
            fs.set_weather(fwi_sel['t2m'].values,
                           rel_hum,
                           fwi_sel['w10'].values,
                           fwi_sel['tp'].values)

            mth, day = fwi_sel['time.month'].values, fwi_sel['time.day'].values
            #print(fs.temp[32,36], fs.prcp[32,36])
            #print(dc0[32,36])
            ffmca = fs.FFMCcalc(ffmc0)
            dmca = fs.DMCcalc(dmc0, mth)
            dca = fs.DCcalc(dc0, mth)
            dcps.append(dca)
            isia = fs.ISIcalc(ffmca)
            buia = fs.BUIcalc(dmca, dca)
            fwia = fs.FWIcalc(isia, buia)
            #print(ffmca[32,36], dmca[32, 36], dca[32,36], fwia[32,36])
            fwis.append(fwia)
            ffmcs.append(ffmca)
            #print(dca[32,36])
            #print(np.any(np.isnan(dca)))
            ffmc0 = ffmca.copy()
            dmc0 = dmca.copy()
            dc0 = dca.copy()
        #print(dcs[0].shape, [x[32,36] for x in dcs])
        dataset = xr.Dataset({'dc': (['time', 'latitude', 'longitude'], dcps),
                              'ffmc': (['time', 'latitude', 'longitude'], ffmcs),
                              'fwi': (['time', 'latitude', 'longitude'], fwis)},
                              coords={'time': times,
                                      'latitude': lats,
                                      'longitude': lons})
        return dataset, ffmc0, dmc0, dc0, dcps


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
        rel_hum = 100 * (top / bot)
        return rel_hum

    def wind_speed(self, dataset):
        wind_speed = np.sqrt(dataset['u10']**2 + dataset['v10']**2)
        return wind_speed

    def update_fwi_vars(self, year, month):
        fname = '/mnt/data/era5/glob/{0}_{1}.nc'.format(year, month)
        fwi_vars = self.prepare_xarray_fwi(fname)
        fwi_vars.to_netcdf('/mnt/data/era5/indonesia/fwi_vars_{0}_{1}.nc'.format(year, month))

    def combine_fwi_vars(self, to_disk = False):
        fnames = glob.glob(self.data_path + '/fwi_vars*')
        dss = []
        for fname in fnames:
            fwi_varr = xr.open_dataset(fname)
            dss.append(fwi_varr)
        dataset = xr.concat(dss, dim = 'time')
        dataset = dataset.sortby('time')
        if to_disk:
            fname = os.path.join(self.data_path, 'combined_fwi_vars.nc')
            if os.path.isfile(fname):
                os.remove(fname)
            dataset.to_netcdf(os.path.join(self.data_path, 'combined_fwi_vars.nc'))
        return dataset

    def daily_climatology(self, dataset, to_disk = False):
        ds_d = dataset.groupby('time.dayofyear').mean(dim = 'time')
        if to_disk:
            ds_d.to_netcdf(os.path.join(self.data_path, 'day_of_year_climatology_fwi_vars.nc'))
        return ds_d

    def monthly_climatology(self, dataset):
        ds_m = dataset.groupby('time.month').mean('time')
        return ds_m

    def era5_climatology(self, start, end):
        ds_clim = xr.open_dataset(os.path.join(self.data_path,
                                               'day_of_year_climatology_fwi_vars.nc'))
        dates = pd.date_range(start, end, freq = 'd')
        clim = ds_clim.sel(dayofyear = dates.dayofyear)
        clim = clim.rename({'dayofyear': 'time'}).assign(time = dates)
        return clim

    def s5_bridge(self, data_path, months_to_bridge):
        dss = []
        for item in months_to_bridge:
             fname = os.path.join(data_path,
                         '{0}_{1:02}/calibrated.nc'.format(item.year, item.month))
             ds = xr.open_dataset(fname)
             date_from = pd.datetime(item.year, item.month, 1) - pd.DateOffset(days = 1)
             date_to = pd.datetime(item.year, item.month, 1) + pd.DateOffset(months = 1)
             ds = ds.sel(time = slice(date_from, date_to))
             fc = self.s5_prepare(ds)
             dss.append(fc)
        return dss

    def s5_prepare(self, fc):
        rel_hum = self.relative_humidity(fc)
        fc = fc.assign({'h2m': rel_hum})
        fc = fc.drop('d2m')
        fc['t2m'] = fc['t2m'] - 273.15
        fc['tp'] = fc['tp'] * 1000
        fc = fc.sortby('latitude', ascending = False)
        return fc

    def s5_forecast(self, data_path, year, month, obs_end_date):
        fname = os.path.join(data_path,
                             '{0}_{1:02}/calibrated.nc'.format(year, month))
        ds = xr.open_dataset(fname)
        fc = self.s5_prepare(ds)
        forecast_start = pd.to_datetime(fc.time[0].values)
        months_to_bridge = pd.date_range(obs_end_date + pd.Timedelta(days = 1),
                                         forecast_start, freq = 'M')
        dss = self.s5_bridge(data_path, months_to_bridge)
        dss.append(fc)
        fc = xr.concat(dss, dim = 'time')
        clim = self.era5_climatology(obs_end_date + pd.DateOffset(1), forecast_start - pd.DateOffset(1))
        fcm = fc.mean(dim = 'number').squeeze()
        fcm = fcm.interp(longitude = clim.longitude, latitude = clim.latitude)
        #s5 = xr.concat([clim, fcm], dim = 'time')
        return fcm

    def stich_era5_and_climatology(self, ds_obs, forecast, month):
        for_start = pd.to_datetime(forecast.time[0].values)
        obs_start = for_start - pd.Timedelta(month, 'M')
        obs = ds_obs.sel(time = slice(obs_start, for_start))
        stiched = xr.concat([obs, forecast], dim = 'time')
        return stiched

    def monthly_obs_all_features_new(self, obs_all):
        obs_m = obs_all[['fwi', 'ffmc', 'dc', 't2m', 'w10', 'h2m']].resample(time = '1M',
                                                                           closed = 'right').median(dim = 'time')
        obs_m = obs_m.rename({'fwi': 'fwi_med', 'ffmc': 'ffmc_med', 'dc': 'dc_med',
                      't2m': 't2m_med', 'w10': 'w10_med', 'h2m': 'h2m_med'})
        obs_mtp = obs_all['tp'].resample(time = '1M', closed = 'right').sum(dim = 'time')
        obs_m['tp_sum'] = obs_mtp
        obs_q = obs_all[['fwi', 'ffmc', 'dc', 't2m', 'w10', 'h2m']].resample(time = '1M',
                                                                             closed = 'right').reduce(np.percentile, q = 75, dim = 'time')
        obs_q = obs_q.rename({'fwi': 'fwi_75p', 'ffmc': 'ffmc_75p', 'dc': 'dc_75p',
                      't2m': 't2m_75p',
                      'w10': 'w10_75p', 'h2m': 'h2m_75p'})
        rolm = obs_all[['fwi', 'ffmc', 'dc', 't2m', 'w10', 'h2m']].rolling(time = 7, min_periods = 1).mean()
        obs_mm = rolm.resample(time = '1M', closed = 'right').max(dim = 'time')
        obs_mm = obs_mm.rename({'fwi': 'fwi_7mm', 'ffmc': 'ffmc_7mm', 'dc': 'dc_7mm',
                      't2m': 't2m_7mm',
                      'w10': 'w10_7mm', 'h2m': 'h2m_7mm'})
        roltp = obs_all['tp'].rolling(time = 7, min_periods = 1).mean()
        obs_mmtp = roltp.resample(time = '1M', closed = 'right').min(dim = 'time')
        obs_mm['tp_7dmin'] = obs_mmtp
        rolm3 = obs_m.rolling(time = 3, min_periods = 1).sum()
        obs_sum3 = rolm3.rename({'fwi_med': 'fwi_3sum', 'ffmc_med': 'ffmc_3sum', 'dc_med': 'dc_3sum',
                      'tp_sum': 'tp_3sum', 't2m_med': 't2m_3sum',
                      'w10_med': 'w10_3sum', 'h2m_med': 'h2m_3sum'})
        obs_m['tp_1'] = obs_mtp.shift(time = 1)
        obs_m['tp_2'] = obs_mtp.shift(time = 2)
        obs_m['tp_3'] = obs_mtp.shift(time = 3)
        obs_m['tp_4'] = obs_mtp.shift(time = 4)
        obs_m['tp_5'] = obs_mtp.shift(time = 5)
        #rolm6 = obs_m.rolling(time = 6, min_periods = 1).sum()
        #obs_sum6 = rolm6.rename({'fwi_med': 'fwi_6sum', 'ffmc_med': 'ffmc_6sum', 'dc_med': 'dc_6sum',
        #              'tp_sum': 'tp_6sum', 't2m_med': 't2m_6sum',
        #              'w10_med': 'w10_6sum', 'h2m_med': 'h2m_6sum'})
        obs_feat = xr.merge([obs_m, obs_q, obs_mm, obs_sum3])
        return obs_feat

    def monthly_fwi_arr_features(self, fwi_arr):
        fwi_m = fwi_arr.resample(time = '1M', closed = 'right').median(dim = 'time')
        fwi_m = fwi_m.rename({'fwi': 'fwi_med', 'ffmc': 'ffmc_med', 'dc': 'dc_med',
                      'tp': 'tp_med', 't2m': 't2m_med',
                      'w10': 'w10_med', 'h2m': 'h2m_med'})
        obs_q = fwi_arr.resample(time = '1M', closed = 'right').reduce(np.percentile, q = 75, dim = 'time')
        fwi_q = obs_q.rename({'fwi': 'fwi_75p', 'ffmc': 'ffmc_75p', 'dc': 'dc_75p',
                      'tp': 'tp_75p', 't2m': 't2m_75p',
                      'w10': 'w10_75p', 'h2m': 'h2m_75p'})
        rolm = fwi_arr.rolling(time = 7, min_periods = 1).mean()
        fwi_mm = rolm.resample(time = '1M', closed = 'right').max(dim = 'time')
        fwi_mm = fwi_mm.rename({'fwi': 'fwi_7mm', 'ffmc': 'ffmc_7mm', 'dc': 'dc_7mm',
                      'tp': 'tp_7mm', 't2m': 't2m_7mm',
                      'w10': 'w10_7mm', 'h2m': 'h2m_7mm'})
        rolm = fwi_arr.rolling(time = 3, min_periods = 1).sum()
        fwi_sum = rolm.resample(time = '1M', closed = 'right').sum(dim = 'time')
        fwi_sum = fwi_sum.rename({'fwi': 'fwi_3sum', 'ffmc': 'ffmc_3sum', 'dc': 'dc_3sum',
                      'tp': 'tp_3sum', 't2m': 't2m_3sum',
                      'w10': 'w10_3sum', 'h2m': 'h2m_3sum'})
        fwi_feat = xr.merge([fwi_m, fwi_q, fwi_mm, fwi_sum])
        return fwi_feat

    ##def monthly_fwi_arr_features(self, fwi_arr):
    #    fwi_m = fwi_arr.resample(time = '1M', closed = 'right').median(dim = 'time')
    #    fwi_m.rename({'fwi': 'fwi_med', 'ffmc': 'ffmc_med', 'dc': 'dc_med'}, inplace = True)
    #    fwi_q = fwi_arr.resample(time = '1M', closed = 'right').reduce(np.percentile, q = 75, dim = 'time')
    #    fwi_q.rename({'fwi': 'fwi_75p', 'ffmc': 'ffmc_75p', 'dc': 'dc_75p'}, inplace = True)
    #    rolm = fwi_arr.rolling(time = 7, min_periods = 1).mean()
    #    fwi_mm = rolm.resample(time = '1M', closed = 'right').max(dim = 'time')
    #    fwi_mm.rename({'fwi': 'fwi_7mm', 'ffmc': 'ffmc_7mm', 'dc': 'dc_7mm'}, inplace = True)
    #    fwi_feat = xr.merge([fwi_m, fwi_q, fwi_mm])
    #    return fwi_feat

if __name__ == '__main__':

    data_path = '/mnt/data/era5/indonesia'
    bbox = [8.0, 93.0, -13.0, 143.0]
    cl = Weather(data_path, bbox = bbox)
    #tpm = get_precipitation(bbox)

    #if new era5 data available:
    #cl.update_fwi_vars(2019, 12)
    #cl.combine_fwi_vars(to_disk = True)

