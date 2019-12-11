import os, glob
import datetime
import h5py
import scipy
#import rasterio
import subprocess
import numpy as np
import xarray as xr
import pandas as pd
from osgeo import gdal, ogr
import matplotlib.pyplot as plt
from pyhdf.SD import SD
import salem
from salem.utils import get_demo_file
from salem import wgs84
#import cartopy.io.shapereader as shapereader
from envdata import Envdata
from gridding import Gridder
from dask.diagnostics import ProgressBar


def calc_area(lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
        """
        # convert decimal degrees to radians 
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        # haversine formula 
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        km = 6371 * c
        return km

def get_tile_ref(fname):
    hv = os.path.basename(fname).split('.')[-4]
    tile_h = int(hv[1:3])
    tile_v = int(hv[4:6])
    return tile_v, tile_h


def plot_dfr_column(gri, dfr, column):
    ds = gri.dfr_to_dataset(dfr, column, np.nan)
    fig = plt.figure(figsize = (12, 5))
    ds[column].plot()
    plt.show()

def plot_dfr_comparison_columns(gri, dfr, dfr2, column, column2):
    ds = gri.dfr_to_dataset(dfr, column, np.nan)
    ds2 = gri.dfr_to_dataset(dfr2, column2, np.nan)
    fig = plt.figure(figsize = (12, 5))
    (ds[column] - ds2[column2]).plot()
    plt.show()


def plot_dfr_comparison_column(gri, dfr, dfr2, column):
    ds = gri.dfr_to_dataset(dfr, column, np.nan)
    ds2 = gri.dfr_to_dataset(dfr2, column, np.nan)
    fig = plt.figure(figsize = (12, 5))
    (ds[column] - ds2[column]).plot()
    plt.show()


def split_loss_per_year():
    data_path = '/mnt/data/forest/loss/'
    fnames = glob.glob(os.path.join(data_path, 'H*.parquet'))
    for year in range(2001, 2019, 1):
        print(year)
        dfrs = []
        for fname in fnames:
            print(fname)
            dfr = pd.read_parquet(fname)
            dfr_year = dfr[dfr.loss_year == int(str(year)[2:])]
            dfrs.append(dfr_year)
        df = pd.concat(dfrs)
        df.reset_index(drop = True, inplace = True)
        df.rename({'loss_year': 'loss'}, axis = 1, inplace = True)
        print(df.head())
        df.to_parquet(os.path.join(data_path, 'loss_{}.parquet'.format(year)))

def loss_primary_or_not(lc):
    data_path = '/mnt/data/forest/primary/'
    fnames = glob.glob(os.path.join(data_path, '*.parquet'))
    for year in range(2001, 2019, 1):
        print(year)
        dfrs = []
        dfr = pd.read_parquet('/mnt/data/forest/loss/loss_{}.parquet'.format(year))
        dfr.sort_values(by = ['latitude', 'longitude'], inplace = True)
        dfr = dfr.drop_duplicates()
        dfr['latitude'] = np.rint(dfr['latitude'] * 1e6).astype(int)
        dfr['longitude'] = np.rint(dfr['longitude'] * 1e6).astype(int)
        for fname in fnames:
            print(fname)
            df = pd.read_parquet(fname)
            dfp = df[df.primary == 2]
            dfp.sort_values(by = ['latitude', 'longitude'], inplace = True)
            dfp['latitude'] = np.rint(dfp['latitude'] * 1e6).astype(int)
            dfp['longitude'] = np.rint(dfp['longitude'] * 1e6).astype(int)
            com = pd.merge(dfr, dfp[['latitude', 'longitude']], how='inner', on=['latitude', 'longitude'])
            dfrs.append(com)
        df = pd.concat(dfrs)
        com = pd.merge(dfr, df, how='left',  on=['latitude', 'longitude'])
        com['loss_type'] = 1
        com['loss_type'][com.loss_y > 0] = 2
        com.drop(['loss_y', 'loss_x'], axis = 1, inplace = True)
        com.to_parquet('/mnt/data/forest/loss/loss_{}_primary_v3.parquet'.format(year))

class LulcData(Envdata):
    def __init__(self, data_path, bbox=None, hour=None):
        self.bbox = bbox
        self.tile_size = 1111950 # height and width of MODIS tile in the projection plane (m)
        self.x_min = -20015109 # the western linit ot the projection plane (m)
        self.y_max = 10007555 # the northern linit ot the projection plane (m)
        self.w_size = 463.31271653 # the actual size of a "500-m" MODIS sinusoidal grid cell
        self.earth_r = 6371007.181 # the radius of the idealized sphere representing the Earth
        super().__init__(data_path, bbox=self.bbox, hour=None)

    def preproces_forest_loss(self):
        fnames = glob.glob(os.path.join(self.data_path, 'H*.tif'))
        for fname in fnames:
            self.preprocess_tiff_to_dfr(fname, 'loss_year')

    def preproces_forest_gain(self):
        #fnames = glob.glob(os.path.join(self.data_path, 'H*.tif'))
        #for fname in fnames:
        #    self.preprocess_tiff_to_dfr(fname, 'gain')
        fnames = glob.glob(os.path.join(self.data_path, 'H*.parquet'))
        self.grid_dfrs(fnames, 0.01, 'gain', '/mnt/data/forest/forest_gain_01deg.parquet')

    def pixel_lon_lat(self, tile_v, tile_h, idi, idj):
        """
        A method to calculate pixel lon lat, using the formulas
        given in the MCD64A1 product ATBD document (Giglio)
        """
        # positions of centres of the grid cells on the global sinusoidal grid

        x_pos = ((idj + 0.5) * self.w_size) + (tile_h * self.tile_size) + self.x_min
        y_pos = self.y_max - ((idi + 0.5) * self.w_size) - (tile_v * self.tile_size)
        # and then lon lat
        lat = y_pos / self.earth_r
        lon = x_pos / (self.earth_r * np.cos(lat))
        return np.rad2deg(lon), np.rad2deg(lat)

    def read_et(self, dataset_path, gri):
        fnames = glob.glob(os.path.join(dataset_path, '*.hdf*'))
        dfrs = []
        for nr, fname in enumerate(fnames):
            ds = self.read_hdf4(fname)
            tile_v, tile_h = get_tile_ref(fname)
            dfr = self.et_to_dataframe(ds, tile_v, tile_h)
            dt_str = fname.split(".")[-5][1:]
            date = pd.to_datetime(dt_str, format = '%Y%j')
            dfr['time'] = date
            dfrs.append(dfr)
        dfr = pd.concat(dfrs)
        return dfr

    def prepare_et_ecosys(self):
        dfr = self.read_et('/mnt/data2/et', gri)
        sp_res = 0.05
        lats = np.arange((-2 + sp_res/2.), 3., sp_res)
        lons = np.arange((99 + sp_res/2.), 104., sp_res)
        gri = Gridder(lats = lats[::-1], lons = lons)
        dfr = gri.add_grid_inds(dfr)
        dfrm = dfr.groupby(['lonind', 'latind', 'time'])['et'].mean()
        dfrm = dfrm.reset_index()
        #dfrm['time'] = pd.datetime(2015, 1, 1)
        dfrm = gri.add_coords_from_ind(dfrm)
        dfrm = self.prepare_ecosys_dataframe(dfrm)
        outn = '/mnt/data2/et/terra_et_raiu_2015_update.csv'
        lc.write_csv(dfrm, outn, fl_prec = '%.3f')
        return dfrm

    def et_to_dataframe(self, ds, tile_v, tile_h):
        et = ds.select('ET_500m').get()
        et_indy, et_indx = np.where(et < 32761)
        et = et[et_indy, et_indx]
        lons, lats = self.pixel_lon_lat(tile_v, tile_h, et_indy, et_indx)
        dfr = pd.DataFrame({'et': et * 0.1,
                            'longitude': lons,
                            'latitude': lats})
        dfr = self.spatial_subset_dfr(dfr, self.bbox)
        return dfr


    def read_ndvi(self, dataset_path, time, sp_res):
        lc_names = ['CMG 0.05 Deg 16 days NDVI']
        lc_data = self.read_hdf4(dataset_path, dataset = lc_names)
        lc_data = np.expand_dims(lc_data[0], axis = 2)
        #lc_data = np.flipud(lc_data)
        lats = np.arange((-90 + sp_res/2.), 90., sp_res)[::-1]
        lons = np.arange((-180 + sp_res/2.), 180., sp_res)
        dataset = xr.Dataset({'ndvi': (['latitude', 'longitude', 'time'], lc_data)},
                              coords={'latitude': lats,
                                     'longitude': lons,
                                      'time': [time]})
        return dataset

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



    def rasterize(self, data_path, input_shpfile, res, bbox = None):
        #image extents are shifted half cell to NE in order to align
        #with ERA5 grid.
        out_file = os.path.join(data_path,
                                os.path.splitext(os.path.basename(input_shpfile))[0] + '_' + str(res))
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
            extents = [bbox[2] - res / 2.,
                       bbox[1] - res / 2.,
                       bbox[3] - res / 2.,
                       bbox[0] - res / 2.]
        tres_str = ' -tr {0} {0}'.format(res)
        tap_str = ' -tap ' + ' '.join(str(x) for x in extents)
        gdal_string = ['gdal_rasterize -a {0} -l {1}'.format(fname, lname) +
                       tres_str, ' -tap ', input_shpfile, out_file + '.tif']
        subprocess.run([' '.join(gdal_string)], shell=True)
        return out_file + '.tif'

    def proc_lulc_shp(self, data_path, input_shpfile, column, res, bbox):
        rast_file = self.rasterize(data_path, input_shpfile, 0.005, bbox = self.bbox)
        dfr_name = self.preprocess_tiff_to_dfr(rast_file, column)
        out_file = os.path.join(data_path,
                os.path.splitext(os.path.basename(input_shpfile))[0] + '_' + str(res) + '.parquet')
        dfr = pd.read_parquet(dfr_name)
        gri = Gridder(bbox = self.bbox, step = res)
        dfr = gri.add_grid_inds(dfr)
        grouped = dfr.groupby(['lonind', 'latind', column]).size().unstack(fill_value = 0)
        grouped.loc[:, 'total'] = grouped.sum(axis = 1)
        grouped.columns = grouped.columns.astype(str)
        grouped.reset_index(inplace = True)
        grouped.to_parquet(out_file)

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

    def preprocess_tiff_to_dfr(self, fname, column):
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
            return None
        dfr.reset_index(inplace=True)
        dfr = dfr.drop('band', axis=1)
        dfr = self.spatial_subset_dfr(dfr, self.bbox)
        dfr.to_parquet(out_name)
        return out_name

    def grid_loss(self, res, column, out_name):
        fnames = glob.glob(os.path.join(self.data_path, 'loss*primary_v3.parquet'))
        print(fnames)
        dfrs = []
        for fname in fnames:
            year = fname.split('_')[-3]
            print(fname)
            dfr = pd.read_parquet(fname)
            print(dfr.columns)
            print(dfr.columns)
            dfr['longitude'] /= 1e6
            dfr['latitude'] /= 1e6
            gri = Gridder(bbox = 'indonesia', step = res)
            dfr = gri.add_grid_inds(dfr)
            print(column)
            grouped = dfr.groupby(['lonind', 'latind', column]).size().unstack(fill_value = 0)
            grouped.rename({1: '{0}_loss'.format(year),
                            2: '{0}_loss_prim'.format(year)},
                           axis = 1, inplace = True)
            grouped.columns = grouped.columns.astype(str)
            grouped.reset_index(inplace = True)
            dfrs.append(grouped)
        grouped = pd.concat(dfrs)
        #grouped = grouped.drop_duplicates()
        grouped = grouped.groupby(['lonind', 'latind']).sum().reset_index()
        #return grouped
        grouped.to_parquet(out_name)


    def grid_dfrs(self, fnames, res, column, out_name):
        print(fnames)
        dfrs = []
        for fname in fnames:
            print(fname)
            dfr = pd.read_parquet(fname)
            print(dfr.columns)
            gri = Gridder(bbox = 'indonesia', step = res)
            dfr = gri.add_grid_inds(dfr)
            print(column)
            grouped = dfr.groupby(['lonind', 'latind', column]).size().unstack(fill_value = 0)
            grouped.loc[:, 'total'] = grouped.sum(axis = 1)
            grouped.columns = grouped.columns.astype(str)
            grouped.reset_index(inplace = True)
            dfrs.append(grouped)
        grouped = pd.concat(dfrs)
        #grouped = grouped.drop_duplicates()
        grouped = grouped.groupby(['lonind', 'latind']).sum().reset_index()
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

    def proc_forest_ds(self, data_path, res):
        for ds_type in ['loss']:
            ds_data_path = os.path.join(data_path, ds_type)
            if ds_type == 'loss':
                fnames = glob.glob(os.path.join(ds_data_path, '*primary.parquet'))
                ds_type = 'loss_type'
            print(fnames)
            out_name = os.path.join(data_path, 'forest_{0}_{1}deg_v2.parquet'.format(ds_type, res))
            self.grid_dfrs(fnames, res, ds_type, out_name)

    def combine_lulcs(self, res):
        prim = pd.read_parquet('/mnt/data/forest/forest_primary_{}deg_clean.parquet'.format(res))
        prim.loc[:, 'f_prim'] = prim['2'] / prim['total']
        prim = prim[['lonind', 'latind', 'total', 'f_prim']]
        loss = pd.read_parquet('/mnt/data/forest/forest_loss_type_{}deg_v2.parquet'.format(res))
        com = pd.merge(prim, loss, how='left', on=['lonind', 'latind'])

        gain = pd.read_parquet('/mnt/data/forest/forest_gain_{}deg_clean.parquet'.format(res))
        gain.rename({'1': 'gain'}, axis=1, inplace=True)
        com = pd.merge(com, gain[['lonind', 'latind', 'gain']], how='left', on=['lonind', 'latind'])

        for i in range(2001, 2018, 1):
            cols = com.filter(regex=str(i))
            for col in cols:
                com.loc[:, col] = com[col] / com['total']
            com.loc[:, ] = com[str(i)] / com['total']

        com['gain'] = com['gain'] / com['total']
        return com

    def combine_forest_dfrs(self):
        prim = pd.read_parquet('/mnt/data/forest/forest_primary_0.25deg_clean.parquet')
        #prim = prim.drop_duplicates()
        #prim = prim.groupby(['lonind', 'latind']).sum().reset_index()
        prim.loc[:, 'f_prim'] = prim['2'] / prim['total']
        prim = prim[['lonind', 'latind', 'total', 'f_prim']]
        prim = prim[prim.total > 700000]

        gain = pd.read_parquet('/mnt/data/forest/forest_gain_0.25deg_v2.parquet')
        gain = gain.drop_duplicates()
        gain = gain.groupby(['lonind', 'latind']).sum().reset_index()

        ds = gri.dfr_to_grid(prim, 'f_prim', np.nan)
        dataset = xr.Dataset({'frp': (['latitude', 'longitude'], np.flipud(ds))},
                              coords={'latitude': gri.lats,
                                     'longitude': gri.lons})

        grid = dataset.salem.grid

        shdf = salem.read_shapefile(get_demo_file('world_borders.shp'))
        ind = shdf[shdf['CNTRY_NAME'] == 'Indonesia']
        sub = dataset.salem.subset(shape=ind, margin=2)

        mask = dataset.salem.roi(shape=ind)
        maska = dataset.salem.roi(shape=ind, all_touched=True)

        #los = pd.read_parquet('/mnt/data/forest/forest_loss_1km.parquet')
        loss = pd.read_parquet('/mnt/data/forest/loss_primary_0.25deg_clean.parquet')
        #loss = loss.drop_duplicates()
        #loss = loss.groupby(['lonind', 'latind']).sum().reset_index()

        #loss.loc[:, 'loss'] = 0
        #loss['loss'][(loss['total'] > 5)] = 1
        #loss = loss[['lonind', 'latind', 'loss']]

        #gain.loc[:, 'gain'] = 1
        gain.rename({'total': 'gain'}, axis = 1, inplace = True)
        gain = gain[['lonind', 'latind', 'gain']]

        tt = pd.merge(prim, loss, on=['lonind', 'latind'])
        ttt = pd.merge(tt, gain, on=['lonind', 'latind'])

        #tfrp = pd.merge(tt, grfrp, on=['lonind', 'latind'])

        gain.loc[:, 'gain'] = 1
        gain = gain[['lonind', 'latind', 'gain']]

        #mask = prim[['lonind', 'latind']].isin({'lonind': loss[loss['loss']==1]['lonind'].values,
        #                                        'latind': loss[loss['loss']==1]['latind'].values}).all(axis=1)
        mask = gain[['lonind', 'latind']].isin({'lonind': prim['lonind'].values,
                                                'latind': prim['latind'].values}).all(axis=1)
        tot = prim[['lonind', 'latind']]

    def create_ecosys_grid_dataset(self, times):
        gri = Gridder(bbox = 'riau', step = 0.05)
        lons = gri.lon_bins[1: -1]
        lats = gri.lat_bins[1:]
        dataset = xr.Dataset({'dummy': (['latitude', 'longitude', 'time'], np.zeros((100, 100, len(times))))},
                              coords={'latitude': lats,
                                     'longitude': lons,
                                     'time' : times})
        return dataset

    def process_soil_moisture_esa_cci(self):
        times = pd.date_range(start='2015-01-01', end='2015-12-31', freq='D')
        fnames = glob.glob('/mnt/data/soil_moisture/C3S*nc')
        dss = []
        lons = pd.read_csv('riau_lons.csv', header = None)[0].values
        lats = pd.read_csv('riau_lats.csv', header = None)[0].values
        gri = Gridder(lons = lons, lats = lats)
        grid_x, grid_y = np.meshgrid(gri.lons, gri.lats)
        for fname in fnames:
            ds = xr.open_dataset(fname)
            ds = ds.rename({'lon': 'longitude', 'lat': 'latitude'})
            ds = self.spatial_subset(ds, [4, 98, -3,  105])
            dfr = ds['sm'].to_dataframe().reset_index()
            sm_grid = scipy.interpolate.griddata((dfr.longitude.values,
                                                  dfr.latitude.values),
                                                 dfr.sm.values, (grid_x, grid_y), method = 'linear')
            sm_grid = np.expand_dims(sm_grid, axis = 2)
            fdate = pd.to_datetime(fname.split('-')[-3][:8])
            dataset = xr.Dataset({'sm': (['latitude', 'longitude', 'time'], sm_grid)},
                              coords={'latitude': gri.lats,
                                     'longitude': gri.lons,
                                     'time' : [fdate]})


            dss.append(dataset)
        dataset = xr.concat(dss, dim = 'time')

        dummy = self.create_ecosys_grid_dataset(times)
        ds = dataset.interp_like(dummy, method = 'nearest')
        dfr = self.prepare_ecosys_dataframe(ds['sm'])
        lc.write_csv(dfr, '/mnt/data/soil_moisture/esa_cci_soil_moisture_riau_2015_0.05deg_linear.csv', fl_prec = '%.3f')

    def process_soil_moisture_SMAP(self):
        fnames = glob.glob('/mnt/data/soil_moisture/SMAP/SMAP_L3*')
        lons = pd.read_csv('riau_lons.csv', header = None)[0].values
        lats = pd.read_csv('riau_lats.csv', header = None)[0].values
        dss = []
        gri = Gridder(lons = lons, lats = lats)
        grid_x, grid_y = np.meshgrid(gri.lons, gri.lats)
        for fname in fnames:
            print(fname)
            fo = h5py.File(fname, 'r')
            lons = fo['Soil_Moisture_Retrieval_Data_AM']['longitude'][()]
            lats = fo['Soil_Moisture_Retrieval_Data_AM']['latitude'][()]
            sm = fo['Soil_Moisture_Retrieval_Data_AM']['soil_moisture'][()]
            inds = np.where((lats < 3.1)&(lats > -3.1)&(lons > 98.9)&(lons < 104.1))
            #loninds = np.where((lons > 98.9)&(lons < 104.1))
            if inds[0].size < 10:
                continue
            lons = lons[inds]
            lats = lats[inds]
            sm = sm[inds]
            sm[sm == -9999] = np.nan
            try:
                sm_grid = scipy.interpolate.griddata((lons,lats), sm, (grid_x, grid_y))
            except:
                continue
            sm_grid = np.expand_dims(sm_grid, axis = 2)
            fdate = pd.to_datetime(fname.split('_')[-3])
            dataset = xr.Dataset({'sm': (['latitude', 'longitude', 'time'], sm_grid)},
                              coords={'latitude': gri.lats,
                                     'longitude': gri.lons,
                                     'time' : [fdate]})
            dss.append(dataset)
        dataset = xr.concat(dss, dim = 'time')
        #dummy = self.create_ecosys_grid_dataset(times)
        #ds = dataset.interp_like(dummy, method = 'nearest')
        dfr = self.prepare_ecosys_dataframe(dataset['sm'])
        lc.write_csv(dfr, '/mnt/data/soil_moisture/SMAP_AM_soil_moisture_riau_2015_0.05deg.csv', fl_prec = '%.3f')
        return dataset

    def prepare_ecosys_dataframe(self, dfr):
        try:
            dfr = dfr.to_dataframe()
        except:
            pass
        dfr.reset_index(inplace=True)
        dfr.loc[:, 'Day'] = dfr['time'].dt.day
        dfr.loc[:, 'Hour'] = dfr['time'].dt.hour
        dfr.loc[:, 'Month'] = dfr['time'].dt.month
        dfr.loc[:, 'Year'] = dfr['time'].dt.year
        dfr.drop('time', axis=1, inplace=True)
        dfr = dfr.dropna(axis = 0)

        dfr = dfr[['latitude', 'longitude', 'Day', 'Hour',
                             'Month', 'Year', 'et']]
        # converting total precipitation to mm from m
        cols = ['lat', 'long', 'Day', 'Hour', 'Month', 'Year', 'ET']
        dfr.columns = cols
        return dfr


    def process_peat(self):
        ds_sum_kal = xr.open_rasterio('/mnt/data/land_cover/peat_depth/WI_peat_atlas/WIpeat_0.01deg.tif')
        ds_papua = xr.open_rasterio('/mnt/data/land_cover/peat_depth/WI_peat_atlas/WIpapua_0.01deg.tif')
        kaldfr = ds_sum_kal.to_dataframe(name = 'peatd').reset_index()
        padfr = ds_papua.to_dataframe(name = 'peatd').reset_index()
        dfr = pd.concat([kaldfr, padfr])
        dfr = dfr.drop('band', axis = 1)
        dfr = dfr[dfr.peatd > 0]
        dfr = dfr.rename({'x': 'longitude', 'y': 'latitude'}, axis = 1)
        dfr['peatd'] = 1
        dfr.to_parquet('/mnt/data/land_cover/peat_depth/peat_mask_0.01deg.parquet')

    def prepare_ndvi_ecosys(self, fname, time):
        nd = lc.read_ndvi(fname, time, 0.05)
        ds = self.spatial_subset(nd, gri.bboxes['riau'])
        #dummy = self.create_ecosys_grid_dataset([time])
        #ds = ndriau.interp_like(dummy, method = 'nearest')
        ds = ds['ndvi'].where(ds['ndvi'] >= 0) * .0001
        ds = ds.fillna(0)
        dfr = self.prepare_ecosys_dataframe(ds)
        outn = '/mnt/data2/ndvi/terra_ndvi_{0}_{1}_{2}.csv'.format(time.year, time.month, time.day)
        lc.write_csv(dfr, outn, fl_prec = '%.3f')


    def process_ndvi(self):
        fn = '/mnt/data2/ndvi/MYD13C1/MYD13C1.A2015185.006.2015303100038.hdf'
        product = SD(fn)


if __name__ == '__main__':
    #data_path = '/mnt/data/land_cover/mcd12c1'
    #data_path = '/mnt/data/land_cover/peatlands'
    #data_path = '/mnt/data/forest/loss'
    data_path = '/mnt/data2/ndvi/MOD13C1'
    #fname = '23_tt_6hourly.nc'
    #fname = 'MCD12C1.A2010001.051.2012264191019.hdf'
    #fname = 'Per-humid_SEA_LC_2015_CRISP_Geotiff_indexed_colour.tif'

    # Riau bbox
    #bbox = [3, 99, -2, 104]

    # Indonesia bbox
    bbox = [8, 93, -13, 143]

    gri = Gridder(bbox = 'riau', step = 0.05)
    lc = LulcData(data_path, bbox=gri.bboxes['riau'], hour=None)
    dfr = lc.prepare_et_ecosys()
    #fname = '/mnt/data2/ndvi/MOD13C1/MOD13C1.A2015193.006.2015304063426.hdf'
    #fname = '/mnt/data2/ndvi/MOD13C1/MOD13C1.A2015209.006.2015304082222.hdf'
    #dt = datetime.datetime(2015, 7, 29)
    #lc.prepare_ndvi_ecosys(fname, dt)




   #ds_all = ds_sum_kal + ds_papua
    #ds_all = ds_all.to_dataset(name = 'depth')
    """
    input_shps = ['/mnt/data/land_cover/peatlands/Peatland_land_cover_1990.shp',
                  '/mnt/data/land_cover/peatlands/Peatland_land_cover_2007.shp',
                  '/mnt/data/land_cover/peatlands/Peatland_land_cover_2015.shp']

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
