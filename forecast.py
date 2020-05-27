"""
SEAS5 data preparation and fire occurence probability modelling routines
"""

import subprocess
import glob, os
import xarray as xr
import pandas as pd
import numpy as np
from gridding import Gridder
from calibrate import Corrector
from fwi_fire import CompData
from fwi_climatology import Weather

class ForecastData():
    def __init__(self, era_data_path, s5_data_path, year, month, res, high_res):
        self.era_data_path = era_data_path
        self.s5_data_path = s5_data_path
        self.year = year
        self.month = month
        self.high_res = high_res
        self.forecast_path = self.make_data_path()
        self.res = res

    def make_data_path(self):
        if self.high_res:
            high_res_path = 'highres'
        else:
            high_res_path = ''
        date_path = '{0}_{1:02}'.format(self.year, self.month)
        forecast_path = os.path.join(self.s5_data_path,
                                     high_res_path,
                                     date_path)
        return forecast_path

    def grib_to_netcdf(self):
        """
        Calls grib_to_netcdf to convert grib_fname file to netcdf
        and writes it to nc_fname
        """
        grib_fname = os.path.join(self.forecast_path,
                                  '{0}_{1:02}_ind.grib'.format(self.year, self.month))
        nc_fname = os.path.join(self.forecast_path,
                                '{0}_{1:02}_ind.nc'.format(self.year, self.month))
        print(grib_fname)
        grib_string = ['grib_to_netcdf ', grib_fname,'-o', nc_fname]
        subprocess.run([' '.join(grib_string)], shell = True)

    def mean_std_biass_adjust(self):
        cr = Corrector(self.era_data_path, self.s5_data_path,
                       self.year, self.month, bbox = 'indonesia', step = 1.)
        s5 = cr.process_forecast(high_res = self.high_res)


    def climatology_daily_ds(self):
        bbox = [8.0, 93.0, -13.0, 143.0]
        wf = Weather(os.path.split(self.era_data_path)[0] + '/indonesia', bbox = bbox)
        with xr.open_dataset(os.path.join(wf.data_path,
                                          'combined_fwi_vars.nc')) as ds_obs:
            from_date = pd.to_datetime(ds_obs.time[-1].values)
        to_date = pd.datetime(self.year, self.month, 1) + pd.tseries.offsets.MonthEnd(1)
        clim_start = from_date + pd.DateOffset(1)
        clim = wf.era5_climatology(clim_start, to_date)
        return clim

    def s5_daily_ds(self):
        bbox = [8.0, 93.0, -13.0, 143.0]
        wf = Weather(os.path.split(self.era_data_path)[0] + '/indonesia', bbox = bbox)
        with xr.open_dataset(os.path.join(wf.data_path,
                                          'combined_fwi_vars.nc')) as ds_obs:
            from_date = pd.to_datetime(ds_obs.time[-1].values)
        s5 = wf.s5_forecast(self.s5_data_path, self.year, self.month, from_date)
        to_date = pd.to_datetime(s5.time[-1].values)
        s5 = s5.where(s5 >= 0, 0)
        clim_start = from_date + pd.DateOffset(1)
        clim = wf.era5_climatology(clim_start, to_date)
        return s5, clim

    def climatology_fwi(self):
        clim = self.climatology_daily_ds()
        ffmc1 = np.load('/mnt/data/fwi/forecast/ffmc1.npy')
        dmc1 = np.load('/mnt/data/fwi/forecast/dmc1.npy')
        dc1 = np.load('/mnt/data/fwi/forecast/dc1.npy')
        wf = Weather(self.era_data_path, bbox = None)
        fwi_arr_clim, ffmc2, dmc2, dc2, dcs2 = wf.calc_fwi(clim, ffmc0 = ffmc1, dmc0 = dmc1, dc0 = dc1)
        clim_all = xr.merge([clim, fwi_arr_clim])
        clim_fname = os.path.join(self.forecast_path, 'clim_all_vars.nc')
        clim_all.to_netcdf(clim_fname)

    def forecast_fwi(self):
        s5, clim = self.s5_daily_ds()
        ffmc1 = np.load('/mnt/data/fwi/forecast/ffmc1.npy')
        dmc1 = np.load('/mnt/data/fwi/forecast/dmc1.npy')
        dc1 = np.load('/mnt/data/fwi/forecast/dc1.npy')
        wf = Weather(self.era_data_path, bbox = None)
        fwi_arr_clim, ffmc2, dmc2, dc2, dcs2 = wf.calc_fwi(clim, ffmc0 = ffmc1, dmc0 = dmc1, dc0 = dc1)
        fwi_arr_s5, ffmc3, dmc3, dc3, dcs3 = wf.calc_fwi(s5, ffmc0 = ffmc1, dmc0 = dmc1, dc0 = dc1)
        clim_all = xr.merge([clim, fwi_arr_clim])
        s5_all = xr.merge([s5, fwi_arr_s5])
        clim_fname = os.path.join(self.forecast_path, 'clim_all_vars.nc')
        s5_fname = os.path.join(self.forecast_path, 's5_all_vars.nc')
        clim_all.to_netcdf(clim_fname)
        s5_all.to_netcdf(s5_fname)

    def clim_monthly_features(self):
        clim_fname = os.path.join(self.forecast_path, 'clim_all_vars.nc')
        clim = xr.open_dataset(clim_fname)
        obs = xr.open_dataset('/mnt/data/fwi/forecast/era5_obs_all_vars.nc')
        self.monthly_features(obs, clim, 'clim')

    def s5_clim_monthly_features(self):
        clim_fname = os.path.join(self.forecast_path, 'clim_all_vars.nc')
        s5_fname = os.path.join(self.forecast_path, 's5_all_vars.nc')
        clim = xr.open_dataset(clim_fname)
        s5 = xr.open_dataset(s5_fname)
        obs = xr.open_dataset('/mnt/data/fwi/forecast/era5_obs_all_vars.nc')
        self.monthly_features(obs, clim, 'clim')
        self.monthly_features(obs, s5, 's5')

    def monthly_features(self, obs, forecast, name):
        wf = Weather(self.era_data_path, bbox = None)
        stiched = wf.stich_era5_and_climatology(obs, forecast, 6)
        stiched_feats = wf.monthly_obs_all_features_new(stiched)
        stiched_feats = stiched_feats.sel(time = slice(pd.datetime(self.year, self.month, 1), None))
        feats_fname = os.path.join(self.forecast_path, '{0}_features.nc'.format(name))
        stiched_feats.to_netcdf(feats_fname)

    def merge_features(self, frp, ds):
        gri = Gridder(bbox = 'indonesia', step = self.res)
        dfr = ds.to_dataframe().reset_index()
        dfr = gri.add_grid_inds(dfr)
        dfr = dfr.rename({'time': 'date'}, axis = 1)
        dfr = dfr.drop(['longitude', 'latitude'], axis = 1)
        merged = pd.merge(frp, dfr, on = ['lonind', 'latind', 'date'], how = 'left')
        return merged

    def merge_frp_weather_features_train(self):
        first_year = 2001
        data_path = '/mnt/data/'
        cc = CompData(data_path, res)
        cc.read_forest_change()
        frp = pd.read_parquet('/mnt/data2/SEAS5/forecast/frp_features_2019_09.parquet')
        frp['date'] = [pd.datetime(2001, 12, 31) + pd.DateOffset(months = x) for x in frp.mind]
        cc.fwi_m = xr.open_dataset('/mnt/data/fwi/forecast/obs_features_tp_new.nc')
        cc.fwi_m = cc.fwi_m.sel(time = slice(None, '2018-12-31'))
        frp_train = frp[frp.date <= cc.fwi_m.time.values[-1]]
        start = pd.Timestamp(self.year, self.month, 1)
        end = start + pd.DateOffset(months = 4)
        cc.fwi_m = cc.fwi_m.sel(time = slice('{0}-01-01'.format(first_year + 1),
                                             '{0}-12-31'.format(self.year)))
        frp = self.merge_features(frp_train, cc.fwi_m)
        frp = frp.drop(['mind'], axis = 1)
        frp.to_parquet('data/feature_train_fr_0.25deg_v4.parquet')

    def merge_frp_weather_features_forecast(self):
        first_year = 2001
        data_path = '/mnt/data/'
        cc = CompData(data_path, res)
        cc.read_forest_change()
        frp = pd.read_parquet('/mnt/data2/SEAS5/forecast/frp_features_2019_10.parquet')
        frp['date'] = [pd.datetime(2001, 12, 31) + pd.DateOffset(months = x) for x in frp.mind]
        clim_fname = os.path.join(self.forecast_path, 'clim_features.nc')
        s5_fname = os.path.join(self.forecast_path, 's5_features.nc')
        cc.fwi_clim = xr.open_dataset(clim_fname)
        cc.fwi_s5 = xr.open_dataset(s5_fname)
        start = pd.Timestamp(self.year, self.month, 1)
        end = start + pd.DateOffset(months = 6)
        frp_fore = frp[frp.date >= cc.fwi_s5.time.values[0]]
        cc.fwi_clim = cc.fwi_clim.sel(time = slice(start, end))
        cc.fwi_s5 = cc.fwi_s5.sel(time = slice(start, end))
        frp_clim = self.merge_features(frp_fore, cc.fwi_clim)
        frp_clim = frp_clim.drop(['mind'], axis = 1)
        frp_clim.to_parquet(os.path.join(self.forecast_path, 'clim_features.parquet'))
        frp_s5 = self.merge_features(frp_fore, cc.fwi_s5)
        frp_s5 = frp_s5.drop(['mind'], axis = 1)
        frp_s5.to_parquet(os.path.join(self.forecast_path, 's5_features.parquet'))

    def merge_frp_weather_features_climatology(self):
        first_year = 2001
        data_path = '/mnt/data/'
        cc = CompData(data_path, res)
        cc.read_forest_change()
        frp = pd.read_parquet('/mnt/data2/SEAS5/forecast/frp_features_2019_10.parquet'.format())
        frp['date'] = [pd.datetime(2001, 12, 31) + pd.DateOffset(months = x) for x in frp.mind]
        clim_fname = os.path.join(self.forecast_path, 'clim_features.nc')
        cc.fwi_clim = xr.open_dataset(clim_fname)
        start = pd.Timestamp(self.year, self.month, 1)
        end = start + pd.DateOffset(months = 6)
        frp_fore = frp[frp.date >= cc.fwi_clim.time.values[0]]
        cc.fwi_clim = cc.fwi_clim.sel(time = slice(start, end))
        frp_clim = self.merge_features(frp_fore, cc.fwi_clim)
        frp_clim = frp_clim.drop(['mind'], axis = 1)
        frp_clim.to_parquet(os.path.join(self.forecast_path, 'clim_features.parquet'))

def era5_fwi():
    era5_path = '/mnt/data/era5/indonesia'
    bbox = [8.0, 93.0, -13.0, 143.0]
    wf = Weather(era5_path, bbox = bbox)
    ds_obs = xr.open_dataset(os.path.join(wf.data_path, 'combined_fwi_vars.nc'))
    ds_obs = ds_obs.sel(time = slice('2001-01-01', ds_obs.time[-1]))
    ffmc0, dmc0, dc0 = None, None, None
    fwi_arr, ffmc1, dmc1, dc1, dcs = wf.calc_fwi(ds_obs, ffmc0 = ffmc0, dmc0 = dmc0, dc0 = dc0)
    obs_all = xr.merge([ds_obs, fwi_arr])
    obs_all.to_netcdf('/mnt/data/fwi/forecast/era5_obs_all_vars.nc')
    np.save('/mnt/data/fwi/forecast/ffmc1', ffmc1)
    np.save('/mnt/data/fwi/forecast/dmc1', dmc1)
    np.save('/mnt/data/fwi/forecast/dc1', dc1)

year = 2019
month = 12
res = 0.25
high_res = False

era_data_path = '/mnt/data/era5/glob'
s5_data_path = '/mnt/data2/SEAS5/forecast'
fd = ForecastData(era_data_path, s5_data_path, year, month, res, high_res = high_res)

#pre step
#if new era5 data available:
#wf.update_fwi_vars(year, month)
#era5_fwi()

#the steps:
fd.grib_to_netcdf()
fd.mean_std_biass_adjust()

#get s5 forecast and climatology datasets and calculate fwi:
fd.forecast_fwi()
#climatology only
#fd.climatology_fwi()

#calculate fwi and weather features
fd.s5_clim_monthly_features()
#clim only
#fd.clim_monthly_features()

#add fire and lc features
fd.merge_frp_weather_features_forecast()
fd.merge_frp_weather_features_climatology()
