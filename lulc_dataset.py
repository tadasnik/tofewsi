import os, glob
import datetime
#import rasterio
import subprocess
import numpy as np
import xarray as xr
import pandas as pd
#from osgeo import gdal, ogr
#from pyhdf.SD import SD
from envdata import Envdata
from gridding import Gridder
from dask.diagnostics import ProgressBar

class LulcData(Envdata):
    def __init__(self, data_path, bbox=None, hour=None):
        self.bbox = bbox
        super().__init__(data_path, bbox=self.bbox, hour=None)

    def read_land_cover(self, dataset_path, sp_res):
        lc_names = ['Majority_Land_Cover_Type_1',
                    'Majority_Land_Cover_Type_2',
                    'Majority_Land_Cover_Type_3']
        lc_data = self.read_hdf4(dataset_path, dataset = lc_names)
        #lc_data = np.flipud(lc_data)
        lats = np.arange((-90 + sp_res/2.), 90., sp_res)[::-1]
        lons = np.arange((-180 + sp_res/2.), 180., sp_res)
        dataset = xr.Dataset({lc_names[0]: (['latitude', 'longitude'], lc_data[0]),
                              lc_names[1]: (['latitude', 'longitude'], lc_data[1]),
                              lc_names[2]: (['latitude', 'longitude'], lc_data[2])},
                              coords={'latitude': lats,
                                     'longitude': lons})
        return dataset

    def preporcess_geotiff_SEA_LULC(self, fname):
        SEA_lulc = {0: 'No class',
                    1: 'Water',
                    2: 'Mangrove',
                    3: 'Peatswamp forest',
                    4: 'Lowland forest',
                    5: 'Lower montane forest',
                    6: 'Upper montane forest',
                    7: 'Regrowth/plantation',
                    8: 'Lowland mosaic',
                    9: 'Montane mosaic',
                    10: 'Lowland open',
                    11: 'Lower montane forest',
                    12: 'Urban',
                    13: 'Large-scale palm'}
        file_name, ext = os.path.splitext(fname)
        out_name = file_name + '.parquet'
        #gdal_string = ['gdal_translate -of netCDF -co "FORMAT=NC4" -a_nodata 0 {0} {1}'.format(fname,
        #                                                                           out_name)]
        #subprocess.run([' '.join(gdal_string)], shell=True)
        ds = xr.open_rasterio(fname)
        ds = ds.rename({'Band1': 'lulc', 'lon': 'longitude', 'lat': 'latitude'})
        lulc = xr.Dataset({'lulc': (['latitude', 'longitude'], np.squeeze(ds.values))},
                coords = {'latitude': ds.y.values, 'longitude': ds.x.values})
        dfr = lulc.to_dataframe()
        dfr = dfr[dfr.lulc != 0]
        dfr.reset_index(inplace = True)
        dfr.to_parquet(out_name)



    def rasterize(self, input_shpfile, resolution, bbox = None):
        #image extents are shifted half cell to NE in order to align
        #with ERA5 grid.
        out_file = os.path.join(self.data_path,
                                os.path.splitext(os.path.basename(input_shpfile))[0] + '_' + str(resolution))
        in_shp = ogr.Open(input_shpfile)
        lyr = in_shp.GetLayer()
        lname = lyr.GetName()
        ldefn = lyr.GetLayerDefn()
        try:
            fname = ldefn.GetFieldDefn(1).name
        except:
            fname = ldefn.GetFieldDefn(0).name
        in_shp = None
        if bbox:
            extents = [bbox[2] - resolution / 2.,
                       bbox[1] - resolution / 2.,
                       bbox[3] - resolution / 2.,
                       bbox[0] - resolution / 2.]
        tres_str = ' -tr {0} {0}'.format(resolution)
        tap_str = ' -tap ' + ' '.join(str(x) for x in extents)
        gdal_string = ['gdal_rasterize -a {0} -l {1}'.format(fname, lname) +
                       tres_str, ' -tap ', input_shpfile, out_file + '.tif']
        subprocess.run([' '.join(gdal_string)], shell=True)
        return out_file + '.tif'

    def subset_dataset(self, dataset):
        if self.bbox:
            dataset = self.spatial_subset(dataset, self.bbox)
        if self.hour:
            dataset = self.time_subset(dataset, self.hour)
        return dataset

    def prepare_dataframe_lc(self, lc_dataset):
        lc_dfr = lc_dataset.to_dataframe()
        lc_dfr.reset_index(inplace=True)
        lc_dfr['Majority_Land_Cover_Type_1'] = lc_dfr['Majority_Land_Cover_Type_1'].astype(int)
        lc_dfr['Majority_Land_Cover_Type_2'] = lc_dfr['Majority_Land_Cover_Type_2'].astype(int)
        lc_dfr['Majority_Land_Cover_Type_3'] = lc_dfr['Majority_Land_Cover_Type_3'].astype(int)
        return lc_dfr

    def write_csv(self, dfr, fname, fl_prec):
        print('writing dataframe to csv file {0}'.format(fname))
        dfr.to_csv(fname, index=False, float_format=fl_prec)
        print('finished writing')

    def lulc_tifs_to_netcdf(self, input_tifs, sp_res=0.05):
        """
        Read lulc rasters in GeoTiff make one xarray dataset and write to netcdf
        """
        x_arrs = []
        ds_names = []
        years = []
        for input_tif in input_tifs:
            base_name = os.path.splitext(os.path.basename(input_tif))[0]
            ds_name = '_'.join(base_name.split('_')[:2])
            year = int(base_name.split('_')[2])
            years.append(year)
            x_arr = xr.open_rasterio(input_tif)
            x_arrs.append(x_arr.values.squeeze())
            lons = x_arr.x.values
            lats = x_arr.y.values
        dataset = xr.Dataset({ds_name: (('year', 'latitude', 'longitude'),
                                                            np.array(x_arrs))},
                              coords = {'year': years,
                                     'latitude': lats,
                                    'longitude': lons})
        dataset[ds_name] = dataset[ds_name].astype(int)
        dataset.to_netcdf(os.path.join(self.data_path, ds_name + '_{0}_deg.nc'.format(sp_res)))
        return dataset

    def preprocess_tiff_to_dfr(self, data_path, column):
        fnames = glob.glob(os.path.join(data_path, '*.tif'))
        dfrs = []
        for fname in fnames:
            print(fname)
            file_name, ext = os.path.splitext(fname)
            out_name = file_name + '.parquet'
            ds = xr.open_rasterio(fname)
            ds = ds.drop('band')
            ds = ds.rename({'x': 'longitude', 'y': 'latitude'})
            dfr = ds.to_dataframe(name=column)
            dfr = dfr[dfr[column] > 0]
            if dfr.empty:
                print('is empty')
                continue
            dfr.reset_index(inplace=True)
            dfr = dfr.drop('band', axis=1)
            dfr = self.spatial_subset_dfr(dfr, self.bbox)
            dfr.to_parquet(out_name)

    def grid_dfrs(self, data_path, column, out_name):
        fnames = glob.glob(os.path.join(data_path, '*.parquet'))
        dfrs = []
        for fname in fnames:
            print(fname)
            dfr = pd.read_parquet(fname)
            gri = Gridder(bbox = self.bbox, step = 0.01)
            dfr = gri.add_grid_inds(dfr)
            grouped = dfr.groupby(['lonind', 'latind', column]).size().unstack(fill_value = 0)
            grouped.loc[:, 'total'] = grouped.sum(axis = 1)
            grouped.columns = grouped.columns.astype(str)
            grouped.reset_index(inplace = True)
            dfrs.append(grouped)
        grouped = pd.concat(dfrs)
        #return grouped
        grouped.to_parquet(out_name)

        """
            gri = Gridder(bbox = self.bbox, step = 0.01)
            dfr = gri.add_grid_inds(dfr)
            grouped = dfr.groupby(['lonind', 'latind', 'primary']).size().unstack(fill_value = 0)
            grouped.loc[:, 'total'] = grouped.sum(axis = 1)
            dfrs.append(grouped)
        grouped = pd.concat(dfrs)
        classes = grouped.columns.values
        print(classes)
        grouped.reset_index(inplace = True)
        dss = []
        for item in classes:
            print(item)
            gridded = self.dfr_to_grid(grouped[['lonind', 'latind', item]], item)
            dataset = xr.Dataset({str(item): (['latitude', 'longitude'], np.flipud(gridded))},
                                  coords={'latitude': self.lats,
                                         'longitude': self.lons})
            dss.append(dataset)
        return xr.merge(dss)
        """

    def combine_forest_dfrs(self):
        prim = pd.read_parquet('/mnt/data/forest/primary/Indonesia_primary_1km.parquet')
        gain = pd.read_parquet('/mnt/data/forest/gain/forest_gain_1km.parquet')
        prim.loc[:, 'prim'] = 0
        prim['prim'][(prim['2'] / prim['total'] > 0.5)] = 1
        prim = prim[['lonind', 'latind', 'prim']]

        loss = pd.read_parquet('/mnt/data/forest/loss/forest_loss_1km.parquet')

        loss.loc[:, 'loss'] = 0
        loss['loss'][(loss['total'] > 5)] = 1
        loss = loss[['lonind', 'latind', 'loss']]

        tt = pd.merge(prim, loss, on=['lonind', 'latind'])
        tfrp = pd.merge(tt, grfrp, on=['lonind', 'latind'])

        gain.loc[:, 'gain'] = 1
        gain = gain[['lonind', 'latind', 'gain']]

        mask = prim[['lonind', 'latind']].isin({'lonind': loss[loss['loss']==1]['lonind'].values,
                                                'latind': loss[loss['loss']==1]['latind'].values}).all(axis=1)
        mask = loss['loss'] == 1][['lonind', 'latind']].isin({'lonind': prim['lonind'].values,
                                                'latind': prim['latind'].values}).all(axis=1)



if __name__ == '__main__':
    #data_path = '/mnt/data/land_cover/mcd12c1'
    #data_path = '/mnt/data/land_cover/peatlands'
    data_path = '/mnt/data/forest/loss'
    #fname = '23_tt_6hourly.nc'
    #fname = 'MCD12C1.A2010001.051.2012264191019.hdf'
    #fname = 'Per-humid_SEA_LC_2015_CRISP_Geotiff_indexed_colour.tif'

    # Riau bbox
    #bbox = [3, 99, -2, 104]

    # Indonesia bbox
    bbox = [8, 93, -13, 143]
    lc = LulcData(data_path, bbox=bbox, hour=None)
    input_shps = ['/mnt/data/land_cover/peatlands/Peatland_land_cover_1990.shp',
                  '/mnt/data/land_cover/peatlands/Peatland_land_cover_2007.shp',
                  '/mnt/data/land_cover/peatlands/Peatland_land_cover_2015.shp']

    """
    with ProgressBar():
        #delayed = df[df['primary'] > 0]
        delayed = df.to_parquet('/mnt/data/forest/primary_forest_2001.parquet')
        results = delayed.compute()
                  '/mnt/data/land_cover/peatlands/Peatland_plantations_1990.shp',
                  '/mnt/data/land_cover/peatlands/Peatland_plantations_2000.shp',
                  '/mnt/data/land_cover/peatlands/Peatland_plantations_2010.shp',
                  '/mnt/data/land_cover/peatlands/Peatland_plantations_2015.shp']
    input_tifs = ['/mnt/data/land_cover/peatlands/Peatland_land_cover_1990.tif',
                  '/mnt/data/land_cover/peatlands/Peatland_land_cover_2007.tif',
                  '/mnt/data/land_cover/peatlands/Peatland_land_cover_2015.tif']
    input_tifs = ['/mnt/data/land_cover/peatlands/Peatland_plantations_1990.tif',
                 '/mnt/data/land_cover/peatlands/Peatland_plantations_2000.tif',
                 '/mnt/data/land_cover/peatlands/Peatland_plantations_2010.tif',
                 '/mnt/data/land_cover/peatlands/Peatland_plantations_2015.tif']
    #dts = ds.lulc_tifs_to_netcdf(input_tifs)

    year = 2011
    data_path = '/home/tadas/tofewsi/data/'
    ds = Climdata(data_path, bbox=bbox, hour=None)
    fc_fname = '{0}-12-31_to_{1}-12-31_169.128_228.128_0.25deg.nc'.format(year-1, year)
    ddd = ds.read_dataset(fc_fname)
    for year in [2010, 2011, 2012, 2013, 2014]:
        an_fname = '{0}-12-31_to_{1}-12-31_165.128_166.128_167.128_168.128_0.25deg.nc'.format(year-1, year)
        fc_fname = '{0}-12-31_to_{1}-12-31_169.128_228.128_0.25deg.nc'.format(year-1, year)
        dfr = ds.prepare_dataframe_era5(an_fname, fc_fname)
        ds.write_csv(dfr, 'era5_{0}_riau.csv'.format(year))


    fpath = '/mnt/data/land_cover/peatlands/Per-humid_SEA_LC_2015_CRISP_Geotiff_indexed_colour.tif'
    fpath = '/mnt/data/land_cover/peatlands/Per-humid_SEA_LC_2015_riau.tif'
    fpath = '/mnt/data/land_cover/peatlands/SEA_LC_2015_riau_0.05.tif'
    df = xr.open_rasterio(fpath)

    dpath = '/mnt/data/land_cover/peatlands/Peatland_land_cover_0.05_deg.nc'
    dts = ds.read_dataset(dpath)
    dts = ds.spatial_subset(dts, bbox)
    for year in [1990, 2007, 2015]:
        dfr = dts.sel(year=year).to_dataframe()
        dfr.reset_index(inplace=True)
        dfr.drop('year', axis=1, inplace=True)
        ds.write_csv(dfr, '/mnt/data/land_cover/peatlands/riau_Peatland_land_cover_{0}.csv'.format(year),
                     fl_prec = '%.3f')
    """
