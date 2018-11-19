import os
import glob
import datetime
import rasterio
import numpy as np
import xarray as xr
import pandas as pd
from envdata import Envdata

class DemData(Envdata):
    def __init__(self, data_path, bbox=None, hour=None):
        super().__init__(data_path, bbox=bbox, hour=None)

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


    def dem_tifs_to_netcdf(self, fnames):
        """
        Read Chirps rasters in GeoTiff make one xarray dataset and write to netcdf
        """
        x_arrs = []
        for nr, fname in enumerate(fnames):
            x_arr = xr.open_rasterio(fname)
            lons = x_arr.x.values
            lats = x_arr.y.values
            darr = xr.DataArray(np.squeeze(x_arr.values), coords=[lats, lons],
                                dims=['latitude', 'longitude'])
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
    data_path = '/mnt/data/dem'
    riau = [3, -2, 99, 104]
    ch = DemData(data_path, bbox=riau)
    #for year in range(2008, 2016, 1):
    #fnames = glob.glob(data_path + '/*.tif')
    fname = os.path.join(data_path, 'out.tif')
    #xars = ch.dem_tifs_to_netcdf(fnames)
    #x_arrs = ch.chirp_tifs_to_netcdf(fnames)
        #ds = xr.concat(x_arrs, dim = 'time')
        #ds = ds.to_dataset('chirps')
        #dfr = ds.to_dataframe()
        #dfr.to_pickle(os.path.join(data_path, 'chirps_{0}'.format(year)))
