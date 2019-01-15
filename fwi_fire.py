import os
import numpy as np
import xarray as xr
import pandas as pd
from envdata import Envdata
#from gridding import Gridder

def spatial_subset_dfr(dfr, bbox):
    """
    Selects data within spatial bbox. bbox coords must be given as
    positive values for the Northern hemisphere, and negative for
    Southern. West and East both positive - Note - the method is
    naive and will only work for bboxes fully fitting in the Eastern hemisphere!!!
    Args:
        dfr - pandas dataframe
        bbox - (list) [North, West, South, East]
    Returns:
        pandas dataframe
    """
    dfr = dfr[(dfr['latitude'] < bbox[0]) &
                            (dfr['latitude'] > bbox[2])]
    dfr = dfr[(dfr['longitude'] > bbox[1]) &
                            (dfr['longitude'] < bbox[3])]
    return dfr

def modis_frp_proc(data_path, years, lats, lons):
    dfrs = []
    gr = Gridder(lats, lons)
    for year in years:
        fname = os.path.join(data_path, 'frp', 'M6_{}.csv'.format(year))
        dfr = pd.read_csv(fname, sep = ',')
        dfr = dfr[['latitude', 'longitude', 'acq_date']]
        dfr.rename({'acq_date': 'date'}, axis = 'columns', inplace = True)
        dfr = spatial_subset_dfr(dfr, [x.values for x in gr.bbox])
        dfrs.append(dfr)
    dfr_all = pd.concat(dfrs, ignore_index = True)
    return dfr_all

class CompData(Envdata):
    def __init__(self, data_path, bbox=None, hour=None):
        super().__init__(data_path, bbox=None, hour=None)
        self.land_mask, self.land_mask_array = self.read_land_mask()

    def read_lulc(self):
        lulc_path = os.path.join(self.data_path, 
                                 'land_cover/peatlands',
                                 'SEA_LC_2015_0.25_dataframe.parquet')
        self.lulc_dfr = pd.read_parquet(lulc_path) 

    def read_land_mask(self):
        land_ds = xr.open_dataset('data/era_land_mask.nc')
        land_ds = self.spatial_subset(land_ds, self.region_bbox['indonesia'])
        return land_ds, np.squeeze(land_ds['lsm'].values)

    def set_frp_ds(self, frp):
        self.frp = frp

    def set_fwi_ds(self, fwi):
        if fwi.time.dt.hour[0] != 0:
            fwi['time'] = fwi['time'] - pd.Timedelta(hours = int(fwi.time.dt.hour[0]))
        self.fwi = fwi

    def temporal_overlap(self):
        if self.frp.time.shape !=  self.fwi.time.shape:
            if self.frp.time.shape > self.fwi.time.shape:
                frp_sub = self.temporal_subset(self.frp, self.fwi)
                self.set_frp_ds(frp_sub)
            else:
                fwi_sub = self.temporal_subset(self.fwi, self.frp)
                self.set_fwi_ds(fwi_sub)

    def temporal_subset(self, large, small):
        large = large.sel(time=slice(small.time[0], small.time[-1]))
        return large

    def compute_monthly(self):
        self.fwi_m = self.fwi.resample(time = '1M', closed = 'right').mean(dim = 'time')
        self.frp_m = self.frp.resample(time = '1M', closed = 'right').sum(dim = 'time')

    def to_land_dfr(self, ds):
        dfr = ds.to_dataframe()
        dfr.reset_index(inplace = True)
        land_dfr = self.land_mask.to_dataframe()
        land_dfr = land_dfr[land_dfr['lsm'] == 1]
        land_dfr.reset_index(inplace = True)
        index1 = pd.MultiIndex.from_arrays([land_dfr[col] for col in ['latitude', 'longitude']])
        index2 = pd.MultiIndex.from_arrays([dfr[col] for col in ['latitude', 'longitude']])
        return dfr.loc[index2.isin(index1)]

    def read_monthly_land_dfr(self):
        self.dfr_m = pd.read_parquet(os.path.join(self.data_path,
                                                  'fwi_frp_monthly_land.parquet'))

    def read_monthly(self):
        self.fwi_m = xr.open_dataset(os.path.join(self.data_path, 'fwi', 'fwi_arr_m.nc'))
        self.frp_m = xr.open_dataset(os.path.join(self.data_path, 'frp', 'frp_count_indonesia_m.nc'))

    def get_pixel(self, lat, lon, fwi_ds):
        frp_pix = self.frp_m['count'].sel(latitude = lat, longitude = lon)
        fwi_pix = self.fwi_m[fwi_ds].sel(latitude = lat, longitude = lon)
        lulc = self.lulc_dfr[(self.lulc_dfr.latitude == lat) &
                             (self.lulc_dfr.longitude == lon)]['lulc']
        return frp_pix, fwi_pix, lulc


if __name__ == '__main__':
    #data_path = '/mnt/data/'
    data_path = '/home/tadas/data/'
    cc = CompData(data_path)
    cc.read_lulc()
    #fwi = xr.open_dataset(os.path.join(data_path, 'fwi', 'fwi_arr.nc'))
    #frp = xr.open_dataset(os.path.join(data_path, 'frp', 'frp_count_indonesia.nc'))
    #cc.set_frp_ds(frp)
    #cc.set_fwi_ds(fwi)
    #cc.temporal_overlap()
    cc.read_monthly()
    cc.read_monthly_land_dfr()
    lulc = os.path.join(data_path, 'land_cover/peatlands/SEA_LC_2015_0.25.tif')
    lulc = xr.open_rasterio(lulc)
    lulc = xr.Dataset({'lulc': (['latitude', 'longitude'], np.squeeze(lulc.values))}, 
            coords = {'latitude': cc.frp_m.latitude, 'longitude': cc.frp_m.longitude})
