import os
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import dask as ds
from dask.distributed import Client
from envdata import Envdata
from fwi.fwi_vectorized import FWI

def calc_fwi(fwi_arr):
    arr_shape = [fwi_arr.dims[x] for x in ['latitude', 'longitude']]
    lats = fwi_arr.latitude.values
    lons = fwi_arr.longitude.values
    times = fwi_arr.time

    #Arrays with initial conditions
    ffmc0 = np.full(arr_shape, 85.0)
    dmc0 = np.full(arr_shape, 6.0)
    dc0 = np.full(arr_shape, 15.0)

    dcs = []
    fwis = []
    #Iterrate over time dimension
    for tt in times:
        print(tt)
        fwi_sel = fwi_arr.sel(time = tt)
        mth, day = fwi_sel['time.month'].values, fwi_sel['time.day'].values
        fs = FWI(fwi_sel['t2m'].values,
                 fwi_sel['h2m'].values,
                 fwi_sel['w10'].values,
                 fwi_sel['tp'].values)
        ffmc = fs.FFMCcalc(ffmc0)
        dmc = fs.DMCcalc(dmc0, mth)
        dc = fs.DCcalc(dc0, mth)
        isi = fs.ISIcalc(ffmc)
        bui = fs.BUIcalc(dmc, dc)
        fwi = fs.FWIcalc(isi, bui)
        ffmc0 = ffmc
        dmc0 = dmc
        dc0 = dc
        dcs.append(dc)
        fwis.append(fwi)

    dataset = xr.Dataset({'dc': (['time', 'latitude', 'longitude'], dcs),
                          'fwi': (['time', 'latitude', 'longitude'], fwis)},
                          coords={'latitude': lats,
                                  'longitude': lons,
                                  'time': times})
    return dataset

class Climdata_dask(Envdata):
    def __init__(self, data_path, bbox=None, hour=None):
        super().__init__(data_path, bbox=bbox, hour=None)

    def dask_client(self):
        self.client = Client(processes=False)

    def read_seas5_dask(self, ds_name):
        dts = xr.open_dataset(ds_name)#, chunks={'number': 1})
        dts = self.spatial_subset(dts, self.bbox)
        return dts

    def prepare_xarray_fwi(self, ds_name):
        fnames = 
        dataset = self.read_seas5_dask(ds_name)
        tp  = dataset['tp'].resample(time='24H',
                                     closed='right',
                                     label='right',
                                     base=7).sum(dim='time')
        tp_diff = np.diff(tp, n=1, axis=0)
        tp_diff = np.concatenate((np.expand_dims(tp[0, :, :, :], axis=0),
                                 tp_diff), axis=0)
        tp.values = tp_diff * 1000
        dataset = dataset[['u10', 'v10', 't2m', 'd2m']]
        noon_ds = dataset.isel({'time': np.where(dataset['time.hour'].isin([6, 12]))[0]})
        rest_ds = noon_ds[['u10', 'v10', 't2m', 'd2m']].resample(time='24H',
                                                                 closed='right',
                                                                 label='right',
                                                                 base=7).max(dim='time')
        w10 = self.wind_speed(rest_ds[['u10', 'v10']])
        w10.name = 'w10'
        h2m = self.relative_humidity(rest_ds[['t2m', 'd2m']])
        h2m.name = 'h2m'
        rest_ds['t2m'] = rest_ds['t2m'] - 273.15
        fwi_vars = xr.merge([rest_ds['t2m'], h2m, w10, tp])
        return fwi_vars

        #converting K to C
        an_dataset['t2m'] = an_dataset['t2m'] - 273.15
        fwi_darray = xr.merge([an_dataset[['t2m', 'w10', 'h2m']], preci[:-1]])
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
    data_path = '/mnt/data/SEAS5'
    ds_name = os.path.join(data_path, '2018_11_seas5.nc')
    # indonesia bbox
    bbox = [8.0, 93.0, -13.0, 143.0]
    cl = Climdata_dask(data_path, bbox=bbox, hour=None)
    client = cl.dask_client()
    fwi_vars = cl.prepare_xarray_fwi(ds_name)
    #fwi_vars.to_netcdf('/mnt/data/SEAS5/2018_11_fwi_vars_indonesia.nc')
    #fwi_arr = xr.open_dataset('/mnt/data/SEAS5/2018_11_fwi_vars_indonesia.nc')
    #dataset.to_netcdf('/mnt/data/SEAS5/fwi/fwi_indonesia_2018_11_{0}.nc'.format(number))
    #dss = []
    #for num in range(0, 51):
    #    ds = calc_fwi(fwi_arr.sel(number=num))
    #    dss.append(ds)
    fwi_arr = xr.open_dataset('/mnt/data/SEAS5/fwi/2018_11_fwi_dc_indonesia.nc')
    land_mask = 'data/era_land_mask.nc'
    land_mask = xr.open_dataset(land_mask)
    land_mask = cl.spatial_subset(land_mask, bbox)
#



