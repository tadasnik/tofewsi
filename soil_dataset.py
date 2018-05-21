import os
import datetime
import numpy as np
import xarray as xr
import pandas as pd
from pyhdf.SD import SD
from envdata import Envdata

class Climdata(Envdata):

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

    def read_geotif(self, sp_res):
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
