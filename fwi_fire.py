import numpy as np
import xarray as xr
import pandas as pd
from envdata import Envdata

class CompData(Envdata):
    def __init__(self, data_path, bbox=None, hour=None):
        super().__init__(data_path, bbox=None, hour=None)
        self.land_mask = self.read_land_mask()

    def read_land_mask(self):
        land_ds = xr.open_dataset('data/era_land_mask.nc')
        land_ds = self.spatial_subset(land_ds, self.region_bbox['indonesia'])
        return land_ds

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

         


if __name__ == '__main__':
    fwi = xr.open_dataset('/home/tadas/data/fwi/fwi_arr.nc')
    frp = xr.open_dataset('/home/tadas/data/frp/frp_count_indonesia.nc')
    cc = CompData('/home/tadas/data')
    cc.set_frp_ds(frp)
    cc.set_fwi_ds(fwi)
    pass



