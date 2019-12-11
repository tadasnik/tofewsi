import os
import glob
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import dask as ds
from dask.distributed import Client
from envdata import Envdata
from gridding import Gridder
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
                       fwi_sel['si10'].values,
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

    def read_seas5_dask(self, ds_name):
        dts = xr.open_dataset(ds_name)#, chunks={'number': 1})
        dts = self.spatial_subset(dts, self.bbox)
        return dts

    def subset_datasets(self, fname):
        gri = Gridder(bbox = 'indonesia', step = 1)
        dss = []
        for fn in fnames:
            ds = xr.open_dataset(fn)
            dsi = self.spatial_subset(ds, gri.bbox)
            #dsi = dsi.median(dim = 'number')
            dss.append(dsi)
        dsa = xr.merge(dss)
        return dsa

    def prepare_s5_fwi(self, dataset):
        h2m = self.relative_humidity(dataset[['t2m', 'd2m']])
        h2m.name = 'h2m'
        #converting K to C
        dataset['t2m'] = dataset['t2m'] - 273.15
        fwi_vars = xr.merge([dataset[['t2m', 'si10', 'tp']], h2m])
        return fwi_vars

    def prepare_era_fwi(self, dataset):
        #dataset = self.read_seas5_dask(ds_name)
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
        si10 = self.wind_speed(rest_ds[['u10', 'v10']])
        si10.name = 'si10'
        h2m = self.relative_humidity(rest_ds[['t2m', 'd2m']])
        h2m.name = 'h2m'
        #converting K to C
        rest_ds['t2m'] = rest_ds['t2m'] - 273.15
        fwi_vars = xr.merge([rest_ds['t2m'], h2m, si10, tp])
        return fwi_vars

        #an_dataset['t2m'] = an_dataset['t2m'] - 273.15
        #fwi_darray = xr.merge([an_dataset[['t2m', 'si10', 'h2m']], preci[:-1]])
        #return fwi_darray

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
    data_path = '/mnt/data2/SEAS5'
    # indonesia bbox
    bbox = [8.0, 93.0, -13.0, 143.0]
    cl = Climdata_dask(data_path, bbox=bbox, hour=None)
    fname = 'corrected_2019_07.nc'
    ds = xr.open_dataset(fname)
    """
    fwi_vars = cl.prepare_s5_fwi(ds)
    #client = cl.dask_client()
    #fnames = glob.glob(os.path.join(data_path, '2019_07_*.nc'))
    #dsa = cl.subset_datasets(fnames)
    #dsa = xr.open_dataset('/mnt/data/SEAS5/2019_03_vars_ind.nc')
    #fwi_vars = cl.prepare_xarray_fwi(fnames)
    #fwi_vars.to_netcdf('/mnt/data/SEAS5/2018_11_fwi_vars_indonesia.nc')
    #fwi_arr = xr.open_dataset('/mnt/data/SEAS5/2018_11_fwi_vars_indonesia.nc')
    #dataset.to_netcdf('/mnt/data/SEAS5/fwi/fwi_indonesia_2018_11_{0}.nc'.format(number))
    dss = []
    for num in range(0, 51):
        ds = calc_fwi(fwi_vars.sel(number=num))
        dss.append(ds[0])
    fwi_ds = xr.concat(dss, dim = 'number')

    #fwi_arr = xr.open_dataset('/mnt/data/SEAS5/fwi/2018_11_fwi_dc_indonesia.nc')
    land_mask = 'data/era_land_mask.nc'
    land_mask = xr.open_dataset(land_mask)
    land_mask = cl.spatial_subset(land_mask, bbox)
    """
