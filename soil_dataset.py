import os
import datetime
import numpy as np
import xarray as xr
import pandas as pd
from pyhdf.SD import SD

class Climdata(object):

    def __init__(self, data_path, bbox=None, hour=None):
        self.data_path = data_path
        self.bbox = bbox
        self.hour = hour
        self.soilgrids_code_names()
        self.soilgrids_levels()

    def soilgrids_code_names(self):
        self.codes_names = {
                'BLDFIE': 'bulk_density',
                'SNDPPT': 'sand_content',
                'SLTPPT': 'silt_content',
                'ORCDRC': 'organic_matter',
                'PHIHOX': 'soil_ph',
                'CECSOL': 'cation_exchange'
                }

    def soilgrids_levels(self):
        self.levels = {
                'sl1': 0,
                'sl2': 5,
                'sl3': 15,
                'sl4': 30,
                'sl5': 60,
                'sl6': 100,
                'sl7': 200
                }

    def read_soil_grids_tiff(self, sp_res):
        """
        Read SoilGrids datasets stored in GeoTiff
        """
        datasets = []
        for prod_key in self.codes_names:
            products = []
            for lev_key in self.levels:
                fname = '{0}_M_{1}_5km_ll.tif'.format(prod_key, lev_key)
                fname_path = os.path.join(self.data_path, fname)
                product = xr.open_rasterio(fname_path)
                products.append(product.values.squeeze())
            lons = product.x.values
            #latitudes need tiying as they are off by a bit
            #porbably due to float conversion somewhere in the pipline
            lats = np.arange(product.y[-1].values.round(0) + sp_res/2, 
                             product.y[0].values + sp_res/2, 
                             sp_res)[::-1]
            dataset = xr.Dataset({self.codes_names[prod_key]: (('level', 'latitude', 'longitude'),
                                                                np.array(products))},
                                  coords = {'level': list(self.levels.values()),
                                         'latitude': lats,
                                        'longitude': lons})
            datasets.append(dataset)
        dataset = xr.merge(datasets)
        return dataset

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

    def subset_dataset(self, dataset):
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
        lat_name = [x for x in list(dataset.coords) if 'lat' in x]
        lon_name = [x for x in list(dataset.coords) if 'lon' in x]
        dataset = dataset.where((dataset[lat_name[0]] <= bbox[0]) &
                                (dataset[lat_name[0]] >= bbox[1]), drop=True)
        dataset = dataset.where((dataset[lon_name[0]] >= bbox[2]) &
                                (dataset[lon_name[0]] <= bbox[3]), drop=True)
        return dataset

    def time_subset(self, dataset, hour=None, start_date=None, end_date=None):
        """
        Selects data for the hour.
        Args:
            dataset - xarray dataset
            hour - (int) hour
        Returns:
            xarray dataset
        """
        if hour:
            dataset = dataset.sel(time=datetime.time(hour))
        return dataset

    def prepare_dataframe_soil(self, soil_dataset):
        for depth in np.unique(soil_dataset.level):
            level_dataset = soil_dataset.where(soil_dataset.level == depth, drop=True)
            lv_dfr = level_dataset.to_dataframe()
            lv_dfr.reset_index(inplace=True)
            lv_dfr.drop('level', axis=1, inplace=True)
            lv_dfr['bulk_density'] = lv_dfr['bulk_density'].astype(int)
            lv_dfr['sand_content'] = lv_dfr['sand_content'].astype(int)
            lv_dfr['silt_content'] = lv_dfr['silt_content'].astype(int)
            lv_dfr['organic_matter'] = lv_dfr['organic_matter'].astype(int)
            lv_dfr['soil_ph'] = lv_dfr['soil_ph'].astype(int)
            lv_dfr['cation_exchange'] = lv_dfr['cation_exchange'].astype(int)
            lv_dfr.replace([255, -32768], -999, inplace=True)
            print(lv_dfr)
            fname_path = os.path.join(self.data_path, 
                                      'soilgrids_riau_depth_{0}_cm.csv'.format(depth))
            self.write_csv(lv_dfr, fname_path, '%.3f')

    def write_csv(self, dfr, fname, fl_prec):
        print('writing dataframe to csv file {0}'.format(fname))
        dfr.to_csv(fname, index=False, float_format=fl_prec)
        print('finished writing')


if __name__ == '__main__':
    #data_path = '/home/tadas/tofewsi/data/'
    #fname = '2013-12-31_to_2014-12-31_169.128_228.128_0.25deg.nc'
    data_path = '/mnt/data/soil/soilgrids/'
    #fname = '23_tt_6hourly.nc'
    fname = 'soilgrids.nc'

    # Riau bbox
    bbox = [3, -2, 99, 104]
    ds = Climdata(data_path, bbox=bbox, hour=None)
    sg = ds.read_dataset(fname)
    #lc = ds.subset_dataset(lc)
    #lc = ds.prepare_dataframe_lc(lc)
    #ds.write_csv(lc, 'lulc_mcd12c1_2010_riau.csv', '%.3f')


    """
    year = 2011
    data_path = '/home/tadas/tofewsi/data/'
    ds = Climdata(data_path, bbox=bbox, hour=None)
    fc_fname = '{0}-12-31_to_{1}-12-31_169.128_228.128_0.25deg.nc'.format(year-1, year)
    ddd = ds.read_dataset(fc_fname)
    """
    """
    for year in [2010, 2011, 2012, 2013, 2014]:
        an_fname = '{0}-12-31_to_{1}-12-31_165.128_166.128_167.128_168.128_0.25deg.nc'.format(year-1, year)
        fc_fname = '{0}-12-31_to_{1}-12-31_169.128_228.128_0.25deg.nc'.format(year-1, year)
        dfr = ds.prepare_dataframe_era5(an_fname, fc_fname)
        ds.write_csv(dfr, 'era5_{0}_riau.csv'.format(year))

    """
