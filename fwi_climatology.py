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

class Climdata_dask(Envdata):
    def __init__(self, data_path, bbox=None, hour=None):
        super().__init__(data_path, bbox=bbox, hour=None)

    def dask_client(self):
        self.client = Client(processes=False)

    def read_era5_dask(self, fname):
        dataset = xr.open_dataset(fname)
        dataset = self.spatial_subset(dataset, self.bbox)
        dataset = self.time_subset(dataset, self.hour)
        return dataset

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
        fwi_darray = xr.merge([an_dataset['t2m'], wind_speed, rel_hum, preci[:-1]])
        return fwi_darray

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


if __name__ == '__main__':

    data_path = '/mnt/data/era5/indonesia'
    #fwi_varr = xr.open_dataset('/mnt/data/era5/indonesia/fwi_vars_{0}_{1}.nc'.format(year, month))
    ffmc0, dmc0, dc0 = None, None, None
    fwi_arrs = []
    for nr, year in enumerate(range(2000, 2019, 1)):
        for month in range(1, 13, 1):
            print(year, month)
            fname = os.path.join(data_path, 'fwi_vars_{0}_{1}.nc'.format(year, month))
            fwi_varr = xr.open_dataset(fname)
            fwi_arr, ffmc0, dmc0, dc0, dcs = calc_fwi(fwi_varr, ffmc0 = ffmc0, dmc0 = dmc0, dc0 = dc0)
            fwi_arrs.append(fwi_arr)
    fwi_ds = xr.concat(fwi_arrs, dim='time')
    """
    month = 1
    data_path = '/mnt/data/era5/glob'
    #fname = os.path.join(data_path, '{0}_{1}.nc'.format(year, month))
    bbox = [8.0, 93.0, -13.0, 143.0]
    cl = Climdata_dask(data_path, bbox = bbox)
    #client = cl.dask_client()
    #initial conditions
    #fwi_vars = cl.prepare_xarray_fwi(fname)
    #fwi_arr, ffmc0, dmc0, dc0 = calc_fwi(fwi_vars)
    fwis = []
    fwivars = []
    for nr, year in enumerate(range(2018, 2019, 1)):
        for month in range(1, 13, 1):
            print(year, month)
            fname = os.path.join(data_path, '{0}_{1}.nc'.format(year, month))
            fwi_vars = cl.prepare_xarray_fwi(fname)
            fwi_vars.to_netcdf('/mnt/data/era5/indonesia/fwi_vars_{0}_{1}.nc'.format(year, month))
            #fwi_arr, ffmc0, dmc0, dc0 = calc_fwi(fwi_vars, ffmc0, dmc0, dc0)
            #fwis.append(fwi_arr)
            fwivars.append(fwi_vars)
    #fwi = xr.concat(fwis, dim='time')
    fwi_var = xr.concat(fwivars, dim='time')
    #fwi_vars.to_netcdf('/mnt/data/SEAS5/2018_11_fwi_vars_indonesia.nc')
    #fwi_arr = xr.open_dataset('/mnt/data/SEAS5/2018_11_fwi_vars_indonesia.nc')
    #dataset.to_netcdf('/mnt/data/SEAS5/fwi/fwi_indonesia_2018_11_{0}.nc'.format(number))
    #dss = []
    #for num in range(0, 51):
    #    ds = calc_fwi(fwi_arr.sel(number=num))
    #    dss.append(ds)
    land_mask = 'data/era_land_mask.nc'
    land_mask = xr.open_dataset(land_mask)
    land_mask = cl.spatial_subset(land_mask, bbox)
    """
