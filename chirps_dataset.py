import os
import glob
import datetime
import rasterio
import numpy as np
import xarray as xr
import pandas as pd
from envdata import Envdata

class ChirpsData(Envdata):
    def __init__(self, data_path, bbox=None, hour=None):
        super().__init__(data_path, bbox=bbox, hour=None)

    def chirp_tifs_to_netcdf(self, input_tifs):
        """
        Read Chirps rasters in GeoTiff make one xarray dataset and write to netcdf
        """
        x_arrs = []
        x_arr = xr.open_rasterio(input_tifs[0])
        lons = x_arr.x.values
        lats = x_arr.y.values
        for nr, input_tif in enumerate(input_tifs):
            base_name = os.path.splitext(os.path.basename(input_tif))[0]
            year, month, day = [int(x) for x in base_name.split('-')]
            datet = pd.datetime(year, month, day)
            x_arr = xr.open_rasterio(input_tif)
            darr = xr.DataArray(x_arr.values, coords=[[datet], lats, lons],
                                dims=['time', 'latitude', 'longitude'])
            darr = self.spatial_subset(darr, self.bbox)
            x_arrs.append(darr)
        #dataset = xr.Dataset({'chirps': (('time', 'latitude', 'longitude'),
        #                                                    np.array(ds.values))},
        #                      coords = {'time': ds.time.values,
        #                             'latitude': ds.latitude,
        #                            'longitude': ds.longitude})
        #dataset[ds_name] = dataset[ds_name].astype(int)
        #dataset.to_netcdf(os.path.join(self.data_path, ds_name + '_{0}_deg.nc'.format(sp_res)))
        return x_arrs


if __name__ == '__main__':
    data_path = '/mnt/data/chirps'
    riau = [3, -2, 99, 104]
    ch = ChirpsData(data_path, bbox=riau)
    #for year in range(2008, 2016, 1):
    year = 2008
    fnames = glob.glob(data_path + '/{0}*.tif'.format(year))
    x_arrs = ch.chirp_tifs_to_netcdf(fnames)
        #ds = xr.concat(x_arrs, dim = 'time')
        #ds = ds.to_dataset('chirps')
        #dfr = ds.to_dataframe()
        #dfr.to_pickle(os.path.join(data_path, 'chirps_{0}'.format(year)))
