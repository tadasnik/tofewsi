import os
import glob
import time
import datetime
import itertools
import numpy as np
import xarray as xr
import pandas as pd
#from sklearn.cluster import DBSCAN
from multiprocessing import Pool, cpu_count
#from gridding import Gridder
#from pyhdf import SD
from envdata import Envdata
import matplotlib.pyplot as plt

def spatial_subset(dataset, bbox):
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
    print(lat_name, bbox)
    dataset = dataset.where((dataset[lat_name[0]] < bbox[0]) &
                            (dataset[lat_name[0]] > bbox[1]), drop=True)
    dataset = dataset.where((dataset[lon_name[0]] > bbox[2]) &
                            (dataset[lon_name[0]] < bbox[3]), drop=True)
    return dataset


def spatial_subset_dfr(dfr, bbox):
    """
    Selects data within spatial bbox. bbox coords must be given as
    positive values for the Northern hemisphere, and negative for
    Southern. West and East both positive - Note - the method is
    naive and will only work for bboxes fully fitting in the Eastern hemisphere!!!
    Args:
        dfr - pandas dataframe
        bbox - (list) [North, South, West, East]
    Returns:
        pandas dataframe
    """
    dfr = dfr[(dfr['lat'] < bbox[0]) &
                            (dfr['lat'] > bbox[1])]
    dfr = dfr[(dfr['lon'] > bbox[2]) &
                            (dfr['lon'] < bbox[3])]
    return dfr

def cluster_haversine(dfr):
    db = DBSCAN(eps=0.8/6371., min_samples=2, algorithm='ball_tree',
            metric='haversine', n_jobs=-1).fit(np.radians(dfr[['lat', 'lon']]))
    return db.labels_

def cluster_euc(xyzt, eps, min_samples):
    #divide space eps by approximate Earth radius in km to get eps in radians.
    #xyzt = np.column_stack((xyz, dates * (eps / time_eps)))
    db = DBSCAN(eps=eps, min_samples=min_samples, algorithm='ball_tree',
            metric='euclidean', n_jobs=-1).fit(xyzt)
    return db.labels_

def lon_lat_to_spherical(dfr):
    lon_rad, lat_rad = np.deg2rad(dfr.lon), np.deg2rad(dfr.lat)
    xyz = spher_to_cartes(lon_rad, lat_rad)
    return xyz

def get_tile_ref(fname):
    hv = os.path.basename(fname).split('.')[-4]
    tile_h = int(hv[1:3])
    tile_v = int(hv[4:6])
    return tile_v, tile_h

def spher_to_cartes(lon_rad, lat_rad):
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.column_stack((x,y,z))

def get_days_since(dfr):
    basedate = pd.Timestamp('2002-01-01')
    dates = pd.to_datetime(dfr.year, format='%Y') + pd.to_timedelta(dfr.date, unit='d')
    dfr.loc[:, 'day_since'] = (dates - basedate).dt.days
    return dfr#(dates - basedate).dt.days

def applyParallel(dfGrouped, func):
    with Pool(cpu_count()) as p:
        ret_list = p.map(func, [group for name, group in dfGrouped])
    return pd.concat(ret_list)

def get_group_rows(df, group_col, condition_col, func, comparison='=='):
     g = df.groupby(group_col)[condition_col]
     condition_limit = g.transform(func)
     return df.query('{0} {1} @condition_limit'.format(condition_col,comparison))

def add_xyz(dfr):
    xyz = lon_lat_to_spherical(dfr)
    dfr.loc[:, 'x'] = xyz[:,0]
    dfr.loc[:, 'y'] = xyz[:,1]
    dfr.loc[:, 'z'] = xyz[:,2]
    return dfr

class FireObs(object):
    def __init__(self, data_path, bbox=None, hour=None):
        self.data_path = data_path
        self.bbox = bbox
        self.hour = hour
        self.tile_size = 1111950 # height and width of MODIS tile in the projection plane (m)
        self.x_min = -20015109 # the western linit ot the projection plane (m)
        self.y_max = 10007555 # the northern linit ot the projection plane (m)
        self.w_size = 463.31271653 # the actual size of a "500-m" MODIS sinusoidal grid cell
        self.earth_r = 6371007.181 # the radius of the idealized sphere representing the Earth
        self.years = list(range(2002, 2016))
        #DBSCAN eps in radians = 650 meters / earth radius
        self.eps = 750 / self.earth_r
        self.basedate = pd.Timestamp('2002-01-01')

        self.labels = ['labs1', 'labs2', 'labs4', 'labs8', 'labs16']

        self.regions_bounds = {'Am_tr': [-113, 31.5, -3.5, -55],
                               'Af_tr': [-18, 22.5, 52, -35],
                               'As_tr': [59.5, 40, 155.5, -30.5]}
        self.africa_block_strings()

    def africa_block_strings(self):
        blocks = itertools.product(['Af_tr'], self.years[:-1])
        self.af_blocks = ['_'.join([x[0], str(x[1])]) for x in blocks]

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

    def read_hdf4(self, file_name, dataset=None):
        """
        Reads Scientific Data Set(s) stored in a HDF-EOS (HDF4) file
        defined by the file_name argument. Returns SDS(s) given
        name string provided by dataset argument. If
        no dataset is given, the function returns pyhdf
        SD instance of the HDF-EOS file open in read mode.
        """
        dataset_path = os.path.join(self.data_path, file_name)
        try:
            product = SD(dataset_path)
            if dataset == 'all':
                dataset = list(product.datasets().keys())
            if isinstance(dataset, list):
                datasetList = []
                for sds in dataset:
                    selection = product.select(sds).get()
                    datasetList.append(selection)
                return datasetList
            elif dataset:
                selection = product.select(dataset).get()
                return selection
            return product
        except IOError as exc:
            print('Could not read dataset {0}'.format(file_name))
            raise

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
        dataset = dataset.where((dataset[lat_name[0]] < bbox[0]) &
                                (dataset[lat_name[0]] > bbox[1]), drop=True)
        dataset = dataset.where((dataset[lon_name[0]] > bbox[2]) &
                                (dataset[lon_name[0]] < bbox[3]), drop=True)
        return dataset

    def time_subset(self, dataset, hour=None, start_date=None, end_date=None):
        """
        Selects data within spatial bbox.
        Args:
            dataset - xarray dataset
            hour - (int) hour
        Returns:
            xarray dataset
        """
        if hour:
            dataset = dataset.sel(time=datetime.time(hour))
        return dataset

    def to_day_since(self, dtime_string):
        """
        Method returning day since the self base date. Takes string datetime in
        YYYY-MM-DD format.
        """
        dtime = pd.to_datetime(dtime_string, format='%Y-%m-%d')
        return (dtime - self.basedate).days


    def get_file_names(self, fname_dir, extension=None):
        if extension:
            pattern = '*.{0}'.format(extension)
        else:
            pattern = '*.*'
        fnames = glob.glob(os.path.join(fname_dir, pattern))
        return fnames

    def check_if_ba(self, ds):
        """
        Check if dataset contains burned pixels
        """
        burned_cells = ds.attributes()['BurnedCells']
        return burned_cells

    def ba_to_dataframe(self, ds, tile_v, tile_h):
        ba_date = ds.select('Burn Date').get()
        ba_indy, ba_indx = np.where(ba_date > 0)
        ba_date = ba_date[ba_indy, ba_indx]
        ba_unc = ds.select('Burn Date Uncertainty').get()[ba_indy, ba_indx]
        ba_qa = ds.select('QA').get()[ba_indy, ba_indx]
        ba_first = ds.select('First Day').get()[ba_indy, ba_indx]
        ba_last = ds.select('Last Day').get()[ba_indy, ba_indx]
        lons, lats = self.pixel_lon_lat(tile_v, tile_h, ba_indy, ba_indx)
        dfr = pd.DataFrame({'date': ba_date,
                            'unc': ba_unc,
                            'qa': ba_qa,
                            #'first_date': ba_first,
                            #'last_date': ba_last,
                            'lon': lons,
                            'lat': lats})
        #selecting only qa == 3 (8 bit field 11000000, 
        #indicating land pixels (first bit) and valid data (second bit) refer to ATBD)
        dfr = dfr[dfr['qa'] == 3]
        dfr.drop('qa', axis=1, inplace=True)
        return dfr

    def read_ba_year(self, year):
        fnames = self.get_file_names(os.path.join(self.data_path, str(year)), 'hdf')
        fr_list = []
        for nr, fname in enumerate(fnames):
            ds = self.read_hdf4(fname)
            burned_cells = ds.attributes()['BurnedCells']
            if burned_cells:
                print(nr)
                tile_v, tile_h = get_tile_ref(fname)
                dfr = self.ba_to_dataframe(ds, tile_v, tile_h)
                fr_list.append(dfr)
        fr_all = pd.concat(fr_list)
        fr_all.reset_index(inplace=True)
        fr_all.loc[:, 'year'] = year
        return fr_all

    def populate_store(self):
        years = list(range(2002, 2016))
        for year in self.years:
            dfr = self.read_ba_year(year)
            dfr.to_hdf(self.store_name, key='ba', format='table',
                       data_columns=['year', 'date', 'lon', 'lat'], append=True)


    def copy_store_tropics(self, store_name, tropics_store_name):
        dfr = pd.read_hdf(self.store_name, 'ba')#, where="year=={0}".format(str(year)))
        for reg in self.regions_bounds.keys():
            if reg == 'Af_tr':
                continue
            bbox = self.regions_bounds[reg]
            sel_dfr = dfr[(dfr.lon > bbox[0])&(dfr.lon < bbox[2])]
            sel_dfr = sel_dfr[(sel_dfr.lat < bbox[1])&(sel_dfr.lat > bbox[3])]
            #sel_dfr = get_days_since(sel_dfr)
            #sel_dfr = add_xyz(sel_dfr)
            #sel_dfr.sort_values(by='day_since', inplace=True)
            parq_name = '{0}.parquet'.format(reg)
            sel_dfr.to_parquet(parq_name, engine='pyarrow')
            #sel_dfr.to_hdf(tropics_store_name, key=reg, format='table', 
            #       data_columns=['year', 'date', 'day_since', 'lon', 'lat'], append=True)

    def preprocess_tropics(self):
        for reg in self.regions_bounds.keys():
            if reg == 'Af_tr':
                for year in self.years[:-1]:
                    lock_strings = self.block_strings(year)
                    dfr = [pd.read_hdf(store_name, 'ba', where=x) for x in block_strings]
                    dfr = pd.concat(dfr)
                    parq_name = '{0}_{1}.parquet'.format(reg, year)
                    self.preprocess(dfr, os.path.join(self.data_path, parq_name))
            else:
                dfr = pd.read_parquet('{0}_unprocessed.parquet'.format(reg))
                parq_name = '{0}.parquet'.format(reg)
                self.preprocess(dfr, os.path.join(self.data_path, '{0}.parquet'))

    def preprocess(self, dfr, out_name):
        dfr = get_days_since(dfr)
        dfr = add_xyz(dfr)
        dfr.sort_values(by='day_since', inplace=True)
        dfr.reset_index(drop=True, inplace=True)
        dfr.to_parquet(out_name, engine='pyarrow')

    def add_labels_to_dfr(self, reg):
        store_name = os.path.join(self.data_path, '{}.parquet'.format(reg))
        dfr = pd.read_parquet(store_name)
        for dur in [1, 2, 4, 8, 16]:
            label_name = os.path.join(self.data_path, '{0}_labels_{1}.parquet'.format(reg, dur))
            labs = pd.read_parquet(label_name)
            dfr.loc[:, 'labs{0}'.format(dur)] = labs.values
        dfr.to_parquet(store_name)

    def populate_store_af_blocks(self, store_name, tropics_store_name):
        for year in self.years[:-1]:
            block_strings = self.block_strings(year)
            dfr = [pd.read_hdf(store_name, 'ba', where=x) for x in block_strings]
            dfr = pd.concat(dfr)
            bbox = self.regions_bounds['Af_tr']
            dfr = dfr[(dfr.lon > bbox[0])&(dfr.lon < bbox[2])]
            dfr = dfr[(dfr.lat < bbox[1])&(dfr.lat > bbox[3])]
            dfr = get_days_since(dfr)
            dfr = add_xyz(dfr)
            dfr.sort_values(by='day_since', inplace=True)
            dfr.reset_index(drop=True, inplace=True)
            parq_name = 'Af_tr_{0}.parquet'.format(year)
            dfr.to_parquet(parq_name, engine='pyarrow')
            #dfr.to_hdf(tropics_store_name, key='Af_tr'+'/block_{0}'.format(year), mode='r+', format='table', 
            #           data_columns=['day_since'], append=True)

    def select_ba(self, selection):
        pass

    def lon_lat_to_spherical(self, dfr):
        lon_rad, lat_rad = np.deg2rad(dfr.lon), np.deg2rad(dfr.lat)
        xyz = spher_to_cartes(lon_rad, lat_rad)
        return xyz

    def write_xyz_day_since(self, store_objects):
        for obj in store_objects:
            print(obj)
            dfr = pd.read_hdf(ba.store_name, obj)
            dfr = add_xyz(dfr)
            dfr[['x', 'y', 'z']].to_hdf(store_name, key='{0}/xyz'.format(obj), append=True)
            #dfr.drop(columns=['x', 'y', 'z'])
            dfr = get_days_since(dfr)
            dfr['day_since'].to_hdf(store_name, key='{0}/day_since'.format(obj), append=True)

    def block_strings(self, year):
        if year not in self.years[:-1]:
            print('year {0} out of range'.format(year))
            return None
        break_day = pd.Timestamp('{0}-04-21'.format(year)).dayofyear
        if year == 2002:
            block_strings = ['year=={0}'.format(year),
                             'year=={0}&date<={1}'.format(year+1, break_day)]
        elif year == 2014:
            block_strings = ['year=={0}&date>{1}'.format(year, break_day),
                             'year=={0}'.format(year+1)]
        else:
            break_day_next = pd.Timestamp('{0}-04-21'.format(year+1)).dayofyear
            block_strings = ['year=={0}&date>{1}'.format(year, break_day),
                             'year=={0}&date<={1}'.format(year+1, break_day_next)]
        return block_strings

    def get_overlap_dur(self, block_string, dur):
        label_str = 'labels_{0}'.format(dur)
        labels_pr = pd.read_hdf(store_name,
                                key=block_string+'/labels_{0}'.format(dur)).values
        dfr_pr = pd.read_hdf(store_name, key=block_string, columns=['lon', 'lat', 'day_since'])
        dfr_pr.loc[:,label_str] = labels_pr
        last_day = dfr_pr['day_since'].max()
        ovarlap_labels = dfr_pr[dfr_pr['day_since'] >= last_day - dur][label_str]
        overlap_dfr = dfr_pr[dfr_pr[label_str].isin(overlap_labels[overlap_labels > -1])]
        overlap_labels = overlap_dfr[label_str]
        #TODO finish this
        #+ -1's from overlap duration and return!

    def cluster_blocks(self, store_name, dur):
        start_time = time.time()
        for dur in np.arange(2, 15, 2):
            print(dur)
            for nr, year in enumerate(self.years[:-1]):
                print(year)
                block_string = 'Af_tr/block_{0}'.format(year)
                dfr = pd.read_hdf(store_name, key=block_string, columns=['x', 'y', 'z', 'day_since'])
                dfr.loc[:, 'day_since_tmp'] = dfr['day_since'] * (self.eps / dur)
                if nr == 0:
                    labels = cluster_euc(dfr[['x', 'y', 'z',
                                                    'day_since_tmp']].values,
                                                                    self.eps,
                                                                    min_samples=2)
                    label_fr = pd.DataFrame({'labels_{0}'.format(dur): labels})
                    label_fr.to_hdf(store_name, key=block_string+'/labels_{0}'.format(dur),
                                    format='table', data_columns=['labels_{0}'.format(dur)],
                                    append=True)
                else:
                    labels = cluster_euc(dfr[['x', 'y', 'z', 'day_since']].values, self.eps, min_samples=2)

                    overlap = get_overlap_dur()
                    label_fr = pd.DataFrame({'labels_{0}'.format(labels): labels})
                    labels_pr = pd.read_hdf(store_name,
                                   key='Af_tr/block_{0}/labels_{1}'.format(years[nr-1], dur))
                fr['labs_{0}'.format(str(dur))] = labels
                fr.to_hdf(store_name, key='{0}/labs_{1}'.format(obj, str(dur)), format='table',
                        columns=['labs_{0}'.format(str(dur))], append=True)
                print("--- %s seconds ---" % (time.time() - start_time))
                start_time = time.time()

    def cluster_region(self, dfr, label_increment=None):
        labels_all = {}
        for dur in [1, 2, 4, 8, 16]:
            print(dur)
            dfr.loc[:, 'day_since_tmp'] = dfr['day_since'] * (self.eps / dur)
            labels = cluster_euc(dfr[['x', 'y', 'z', 'day_since_tmp']].values, self.eps, min_samples=1)
            if label_increment:
                labels += label_increment['labs{0}_inc'.format(dur)]
            labels_all['labs{0}'.format(dur)] = labels
        labels_fr = pd.DataFrame(labels_all)
        return labels_fr

    def write_dfr_to_parquet(self, dfr, region_id):
        store_name = os.path.join(self.data_path, '{0}.parquet'.format(region_id))
        dfr.to_parquet(store_name)

    def read_dfr_from_parquet(self, region_id, columns = None):
        store_name = os.path.join(self.data_path, '{0}.parquet'.format(region_id))
        dfr = pd.read_parquet(store_name, columns=columns)
        return dfr

    def get_label_increment(self, reg_id):
        dfr = self.read_dfr_from_parquet(reg_id, columns = self.labels)
        incd = {}
        for dur in [1, 2, 4, 8, 16]:
            incd['labs{0}_inc'.format(dur)] = dfr['labs{0}'.format(dur)].max() + 1
        return incd

    def append_overlap(self, dfr, overlap):
        overlap = self.clean_dfr(overlap)
        try:
            overlap.drop(self.labels, axis=1, inplace = True)
        except:
            pass
        dfr = dfr.append(overlap, ignore_index = True)
        dfr.sort_values(['day_since'], inplace = True)
        dfr.reset_index(drop = True, inplace = True)
        return dfr


    def clean_dfr(self, dfr):
        try:
            dfr.drop(['index'],axis=1, inplace=True)
        except:
            pass
        try:
            dfr.drop(['level_0'],axis=1, inplace=True)
        except:
            pass
        try:
            dfr.drop(['day_since_tmp'],axis=1, inplace=True)
        except:
            pass
        return dfr

    def cluster_Af_blocks(self):
        for nr, block in enumerate(self.years[:-1]):
            print(block)
            if nr == 0:
                dfr = self.read_dfr_from_parquet('Af_tr_{0}'.format(block))
                label_increment = None
            else:
                dfr = self.read_dfr_from_parquet('Af_tr_{0}_mod'.format(block))
                prev_region_name = 'Af_tr_{0}_mod_lab'.format(self.years[nr-1])
                label_increment = self.get_label_increment(prev_region_name)
            label_fr = self.cluster_region(dfr, label_increment)
            dfr = pd.concat([dfr, label_fr], axis=1)
            if block == self.years[-2]:
                 self.write_dfr_to_parquet('Af_tr_{0}_mod_lab'.format(block))
                 break
            #dfr.to_parquet('Af_tr' + '_{0}_'.format(block))
            labs_last = dfr[dfr['day_since'] >= dfr['day_since'].max()-16]['labs16'].unique()
            overlap = dfr[dfr['labs16'].isin(labs_last)]
            self.write_dfr_to_parquet(dfr[~dfr['labs16'].isin(labs_last)],
                                      'Af_tr_{0}_mod_lab'.format(block))
            next_region_name = 'Af_tr_{0}'.format(self.years[nr+1])
            dfr_next = self.read_dfr_from_parquet(next_region_name)
            dfr_next = self.clean_dfr(dfr_next)
            dfr_next = self.append_overlap(dfr_next, overlap)
            self.write_dfr_to_parquet(dfr_next, next_region_name+'_mod')

    def clustering(self, reg):
        if reg == 'Af_tr':
            self.cluster_Af_blocks()
        else:
            region_name = 'data/{0}.parquet'.format(reg)
            dfr = self.read_dfr_from_parquet(region_name, columns = ['x', 'y', 'z', 'day_since'])
            self.cluster_region(dfr)
            self.add_labels_to_dfr(region_name)

    def centroids_pandas(self, parq_name, dur):
        store_name = os.path.join(self.data_path, '{0}.parquet'.format(parq_name))
        dfr = self.read_dfr_from_parquet(parq_name, columns=['lon',
                                                             'lat',
                                                             'day_since',
                                                             'labs1',
                                                             'labs{0}'.format(dur)])
        dates = pd.date_range('2002-01-01', periods=dfr.day_since.max(), freq='d')
        gr = dfr.groupby(['labs{0}'.format(dur)])['day_since']
        condition_limit = gr.transform(min)
        reduced_dfr = dfr.query('day_since == @condition_limit')
        centroids = reduced_dfr.groupby(['labs1', 'day_since']).agg({'lon':'mean', 'lat':'mean'})
        centroids.reset_index(level=1, inplace=True)
        centroids.reset_index(drop=True, inplace=True)
        centroids.loc[:, 'date'] = dates[centroids.day_since-1]
        centroids.loc[:, 'year'] = centroids.date.dt.year
        return centroids

    def combine_centroids(self, dur):
        regions = ['As_tr', 'Am_tr']
        regions.extend(self.af_blocks)
        parq_names = ['{0}_mod_lab'.format(x) for x in regions]
        centroids = [self.centroids_pandas(x, dur) for x in parq_names]
        centroids = pd.concat(centroids, ignore_index=True)
        return centroids

    def annual_averages(self):
        """
        A method to derive annual averages of a daily ignitions grids
        """
        for year in self.years:
            fname = 'ignitions_tropics_{0}.nc'.format(year)
            ds = xr.open_dataset(os.path.join(self.data_path, fname))
            an_mean = ds.sum(dim='date')
            an_mean.to_netcdf(os.path.join(self.data_path,
                              'ignitions_yearly_sum_{0}.nc'.format(year)))



def ds_monthly_means_2d(darray, land_mask):
    darray_m = darray.groupby('time.month').mean('time') 
    darray_masked = darray_m.where(land_mask.values)
    return darray_masked

def ds_monthly_means(darray, land_mask):
    darray_m = darray.groupby('time.month').mean() 
    darray_masked = darray_m.where(land_mask.values)
    return darray_masked


def dfr_monthly_counts(dfr):
    dfr_m = dfr.day_since.groupby([dfr.date.dt.year, 
                                   dfr.date.dt.month]).count().mean(level=1)
    return dfr_m 


def plot_comp_gfas(fwi, gfas, bboxes, land_mask, y2_label):
    fig = plt.figure(figsize=(19,10))

    fwi15 = fwi.sel(time = '2015')
    ba15 = gfas.sel(time = '2015')

    fwi = fwi.sel(time = fwi['time.year'] < 2015)
    ba = gfas.sel(time = gfas['time.year'] < 2015)

    fwi_m = ds_monthly_means(spatial_subset(fwi, bboxes['Indonesia']), land_mask)
    ba_m = ds_monthly_means(spatial_subset(ba, bboxes['Indonesia']), land_mask)
    ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=1)
    months = list(range(1, 13, 1))
    ax1.plot(months, fwi_m.values, 'b-')
    ax1.set_xlabel('Month')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Mean FWI 2008 - 2014' , color='b')
    ax1.tick_params('y', colors='b')
    ax12 = ax1.twinx()
    ax12.set_ylabel('Mean ' + y2_label + ' 2008 - 2014', color='r')
    print(ba_m)
    ax12.bar(months, ba_m, color='r', alpha=.6)
    ax12.tick_params('y', colors='r')
    ax1.set_title(list(bboxes.keys())[0])

    ax2 = plt.subplot2grid((2, 4), (0, 1), colspan=1)
    fwi_m = ds_monthly_means(spatial_subset(fwi, bboxes['Kalimantan']), land_mask)
    ba_m = ds_monthly_means(spatial_subset(ba, bboxes['Kalimantan']), land_mask)
    print(bboxes['Kalimantan'])
    ax2.plot(months, fwi_m.values, 'b-')
    ax2.set_xlabel('Month')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax2.set_ylabel('Mean FWI 2008 - 2014', color='b')
    ax2.tick_params('y', colors='b')
    ax22 = ax2.twinx()
    ax22.bar(months, ba_m, color='r', alpha=.6)
    ax22.set_ylabel('Mean ' + y2_label + ' 2008 - 2014', color='r')
    ax22.tick_params('y', colors='r')
    ax2.set_title(list(bboxes.keys())[1])


    ax3 = plt.subplot2grid((2, 4), (0, 2), colspan=1)
    fwi_m = ds_monthly_means(spatial_subset(fwi, bboxes['South Sumatra']), land_mask)
    ba_m = ds_monthly_means(spatial_subset(ba, bboxes['South Sumatra']), land_mask)
    print(bboxes['South Sumatra'])
    ax3.plot(months, fwi_m.values, 'b-')
    ax3.set_xlabel('Month')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax3.set_ylabel('Mean FWI 2008 - 2014', color='b')
    ax3.tick_params('y', colors='b')
    ax32 = ax3.twinx()
    ax32.bar(months, ba_m, color='r', alpha=.6)
    ax32.set_ylabel('Mean ' + y2_label+ ' 2008 - 2014', color='r')
    ax32.tick_params('y', colors='r')
    ax3.set_title(list(bboxes.keys())[2])

    ax4 = plt.subplot2grid((2, 4), (0, 3), colspan=1)
    fwi_m = ds_monthly_means(spatial_subset(fwi, bboxes['Inner Riau']), land_mask)
    ba_m = ds_monthly_means(spatial_subset(ba, bboxes['Inner Riau']), land_mask)
    ax4.plot(months, fwi_m.values, 'b-')
    ax4.set_xlabel('Month')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax4.set_ylabel('Mean FWI 2008 - 2014', color='b')
    ax4.tick_params('y', colors='b')
    ax42 = ax4.twinx()
    ax42.bar(months, ba_m, color='r', alpha=.6)
    ax42.set_ylabel('Mean ' + y2_label+ ' 2008 - 2014', color='r')
    ax42.tick_params('y', colors='r')
    ax4.set_title(list(bboxes.keys())[3])

    #2015
    fwi_m = ds_monthly_means(spatial_subset(fwi15, bboxes['Indonesia']), land_mask)
    ba_m = ds_monthly_means(spatial_subset(ba15, bboxes['Indonesia']), land_mask)
    ax11 = plt.subplot2grid((2, 4), (1, 0), colspan=1)
    months = list(range(1, 13, 1))
    ax11.plot(months, fwi_m.values, 'b-')
    ax11.set_xlabel('Month')
    # Make the y-ax1is label, ticks and tick labels match the line color.
    ax11.set_ylabel('Mean FWI 2015', color='b')
    ax11.tick_params('y', colors='b')
    ax112 = ax11.twinx()
    ax112.set_ylabel(y2_label + ' 2015', color='r')
    ax112.bar(months, ba_m, color='r', alpha=.6)
    ax112.tick_params('y', colors='r')
    ax11.set_title(list(bboxes.keys())[0])

    ax12 = plt.subplot2grid((2, 4), (1, 1), colspan=1)
    fwi_m = ds_monthly_means(spatial_subset(fwi15, bboxes['Kalimantan']), land_mask)
    ba_m = ds_monthly_means(spatial_subset(ba15, bboxes['Kalimantan']), land_mask)
    print(bboxes['Kalimantan'])
    ax12.plot(months, fwi_m.values, 'b-')
    ax12.set_xlabel('Month')
    # Make the y-ax1is label, ticks and tick labels match the line color.
    ax12.set_ylabel('Mean FWI 2015', color='b')
    ax12.tick_params('y', colors='b')
    ax122 = ax12.twinx()
    ax122.bar(months, ba_m, color='r', alpha=.6)
    ax122.set_ylabel(y2_label + ' 2015', color='r')
    ax122.tick_params('y', colors='r')
    ax12.set_title(list(bboxes.keys())[1])


    ax13 = plt.subplot2grid((2, 4), (1, 2), colspan=1)
    fwi_m = ds_monthly_means(spatial_subset(fwi15, bboxes['South Sumatra']), land_mask)
    ba_m = ds_monthly_means(spatial_subset(ba15, bboxes['South Sumatra']), land_mask)
    print(bboxes['South Sumatra'])
    ax13.plot(months, fwi_m.values, 'b-')
    ax13.set_xlabel('Month')
    # Make the y-ax1is label, ticks and tick labels match the line color.
    ax13.set_ylabel('Mean FWI 2015', color='b')
    ax13.tick_params('y', colors='b')
    ax132 = ax13.twinx()
    ax132.bar(months, ba_m, color='r', alpha=.6)
    ax132.set_ylabel(y2_label + ' 2015', color='r')
    ax132.tick_params('y', colors='r')
    ax13.set_title(list(bboxes.keys())[2])

    ax14 = plt.subplot2grid((2, 4), (1, 3), colspan=1)
    fwi_m = ds_monthly_means(spatial_subset(fwi15, bboxes['Inner Riau']), land_mask)
    ba_m = ds_monthly_means(spatial_subset(ba15, bboxes['Inner Riau']), land_mask)
    ax14.plot(months, fwi_m.values, 'b-')
    ax14.set_xlabel('Month')
    # Make the y-ax1is label, ticks and tick labels match the line color.
    ax14.set_ylabel('Mean FWI 2015', color='b')
    ax14.tick_params('y', colors='b')
    ax142 = ax14.twinx()
    ax142.bar(months, ba_m, color='r', alpha=.6)
    ax142.set_ylabel(y2_label + ' 2015', color='r')
    ax142.tick_params('y', colors='r')
    ax14.set_title(list(bboxes.keys())[3])


    fig.suptitle('GFAS FRP density', size=16)
    #fig.suptitle('MODIS FRP Collection 6', size=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('GFAS_FRP_density.png', res=300)
    plt.show()


def plot_comp(fwi, ba, frp, gfas, bboxes, land_mask, y2_label):
    fig = plt.figure(figsize=(19,10))

    fwi15 = fwi.sel(time = '2015')
    ba15 = ba[ba.date.dt.year == 2015]

    fwi = fwi.sel(time = fwi['time.year'] < 2015)
    ba = ba[ba.date.dt.year < 2015]

    fwi_m = ds_monthly_means(spatial_subset(fwi, bboxes['Indonesia']), land_mask)
    ba_m = dfr_monthly_counts(spatial_subset_dfr(ba, bboxes['Indonesia']))
    ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=1)
    months = list(range(1, 13, 1))
    ax1.plot(months, fwi_m.values, 'b-')
    ax1.set_xlabel('Month')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Mean FWI 2008 - 2014' , color='b')
    ax1.tick_params('y', colors='b')
    ax12 = ax1.twinx()
    ax12.set_ylabel('Mean ' + y2_label + ' 2008 - 2014', color='r')
    ax12.bar(ba_m.index.values, ba_m, color='r', alpha=.6)
    ax12.tick_params('y', colors='r')
    ax1.set_title(list(bboxes.keys())[0])

    ax2 = plt.subplot2grid((2, 4), (0, 1), colspan=1)
    fwi_m = ds_monthly_means(spatial_subset(fwi, bboxes['Kalimantan']), land_mask)
    ba_m = dfr_monthly_counts(spatial_subset_dfr(ba, bboxes['Kalimantan']))
    print(bboxes['Kalimantan'])
    ax2.plot(months, fwi_m.values, 'b-')
    ax2.set_xlabel('Month')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax2.set_ylabel('Mean FWI 2008 - 2014', color='b')
    ax2.tick_params('y', colors='b')
    ax22 = ax2.twinx()
    ax22.bar(ba_m.index.values, ba_m, color='r', alpha=.6)
    ax22.set_ylabel('Mean ' + y2_label + ' 2008 - 2014', color='r')
    ax22.tick_params('y', colors='r')
    ax2.set_title(list(bboxes.keys())[1])


    ax3 = plt.subplot2grid((2, 4), (0, 2), colspan=1)
    fwi_m = ds_monthly_means(spatial_subset(fwi, bboxes['South Sumatra']), land_mask)
    ba_m = dfr_monthly_counts(spatial_subset_dfr(ba, bboxes['South Sumatra']))
    print(bboxes['South Sumatra'])
    ax3.plot(months, fwi_m.values, 'b-')
    ax3.set_xlabel('Month')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax3.set_ylabel('Mean FWI 2008 - 2014', color='b')
    ax3.tick_params('y', colors='b')
    ax32 = ax3.twinx()
    ax32.bar(ba_m.index.values, ba_m, color='r', alpha=.6)
    ax32.set_ylabel('Mean ' + y2_label+ ' 2008 - 2014', color='r')
    ax32.tick_params('y', colors='r')
    ax3.set_title(list(bboxes.keys())[2])

    ax4 = plt.subplot2grid((2, 4), (0, 3), colspan=1)
    fwi_m = ds_monthly_means(spatial_subset(fwi, bboxes['Inner Riau']), land_mask)
    ba_m = dfr_monthly_counts(spatial_subset_dfr(ba, bboxes['Inner Riau']))
    ax4.plot(months, fwi_m.values, 'b-')
    ax4.set_xlabel('Month')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax4.set_ylabel('Mean FWI 2008 - 2014', color='b')
    ax4.tick_params('y', colors='b')
    ax42 = ax4.twinx()
    ax42.bar(ba_m.index.values, ba_m, color='r', alpha=.6)
    ax42.set_ylabel('Mean ' + y2_label+ ' 2008 - 2014', color='r')
    ax42.tick_params('y', colors='r')
    ax4.set_title(list(bboxes.keys())[3])

    #2015
    fwi_m = ds_monthly_means(spatial_subset(fwi15, bboxes['Indonesia']), land_mask)
    ba_m = dfr_monthly_counts(spatial_subset_dfr(ba15, bboxes['Indonesia']))
    ax11 = plt.subplot2grid((2, 4), (1, 0), colspan=1)
    months = list(range(1, 13, 1))
    ax11.plot(months, fwi_m.values, 'b-')
    ax11.set_xlabel('Month')
    # Make the y-ax1is label, ticks and tick labels match the line color.
    ax11.set_ylabel('Mean FWI 2015', color='b')
    ax11.tick_params('y', colors='b')
    ax112 = ax11.twinx()
    ax112.set_ylabel(y2_label + ' 2015', color='r')
    ax112.bar(ba_m.index.values, ba_m, color='r', alpha=.6)
    ax112.tick_params('y', colors='r')
    ax11.set_title(list(bboxes.keys())[0])

    ax12 = plt.subplot2grid((2, 4), (1, 1), colspan=1)
    fwi_m = ds_monthly_means(spatial_subset(fwi15, bboxes['Kalimantan']), land_mask)
    ba_m = dfr_monthly_counts(spatial_subset_dfr(ba15, bboxes['Kalimantan']))
    print(bboxes['Kalimantan'])
    ax12.plot(months, fwi_m.values, 'b-')
    ax12.set_xlabel('Month')
    # Make the y-ax1is label, ticks and tick labels match the line color.
    ax12.set_ylabel('Mean FWI 2015', color='b')
    ax12.tick_params('y', colors='b')
    ax122 = ax12.twinx()
    ax122.bar(ba_m.index.values, ba_m, color='r', alpha=.6)
    ax122.set_ylabel(y2_label + ' 2015', color='r')
    ax122.tick_params('y', colors='r')
    ax12.set_title(list(bboxes.keys())[1])


    ax13 = plt.subplot2grid((2, 4), (1, 2), colspan=1)
    fwi_m = ds_monthly_means(spatial_subset(fwi15, bboxes['South Sumatra']), land_mask)
    ba_m = dfr_monthly_counts(spatial_subset_dfr(ba15, bboxes['South Sumatra']))
    print(bboxes['South Sumatra'])
    ax13.plot(months, fwi_m.values, 'b-')
    ax13.set_xlabel('Month')
    # Make the y-ax1is label, ticks and tick labels match the line color.
    ax13.set_ylabel('Mean FWI 2015', color='b')
    ax13.tick_params('y', colors='b')
    ax132 = ax13.twinx()
    ax132.bar(ba_m.index.values, ba_m, color='r', alpha=.6)
    ax132.set_ylabel(y2_label + ' 2015', color='r')
    ax132.tick_params('y', colors='r')
    ax13.set_title(list(bboxes.keys())[2])

    ax14 = plt.subplot2grid((2, 4), (1, 3), colspan=1)
    fwi_m = ds_monthly_means(spatial_subset(fwi15, bboxes['Inner Riau']), land_mask)
    ba_m = dfr_monthly_counts(spatial_subset_dfr(ba15, bboxes['Inner Riau']))
    ax14.plot(months, fwi_m.values, 'b-')
    ax14.set_xlabel('Month')
    # Make the y-ax1is label, ticks and tick labels match the line color.
    ax14.set_ylabel('Mean FWI 2015', color='b')
    ax14.tick_params('y', colors='b')
    ax142 = ax14.twinx()
    ax142.bar(ba_m.index.values, ba_m, color='r', alpha=.6)
    ax142.set_ylabel(y2_label + ' 2015', color='r')
    ax142.tick_params('y', colors='r')
    ax14.set_title(list(bboxes.keys())[3])


    fig.suptitle('MCD64A1 Collection 6 BA', size=16)
    #fig.suptitle('MODIS FRP Collection 6', size=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('MCD64_BA.png', res=300)
    plt.show()




if __name__ == '__main__':
    indonesia_bbox = [7.0, -11.0, 93.0, 143.0]
    kalimantan = [7.0, -4.5, 108.0, 119]
    sumatra_south = [3, -6, 98, 106]
    riau_inner = [1,  -0.4, 101, 103.5]
    bboxes = {'Indonesia': indonesia_bbox,
              'Kalimantan': kalimantan,
              'South Sumatra': sumatra_south,
              'Inner Riau': riau_inner}


    pass
"""
    data_path = '~/data/'
    land_mask = '~/data/land_mask/land_mask_indonesia.nc'
    fwi_ds = '~/data/fwi/fwi_dc_indonesia.nc'
    ba_prod = '~/data/ba/indonesia_ba.parquet'


    fo = FireObs(data_path)
    #orig_indonesia_bbox = [8.0, -13.0, 93.0, 143.0]
    indonesia_bbox = [7.0, -11.0, 93.0, 143.0]
    kalimantan = [7.0, -4.5, 108.0, 119]
    sumatra_south = [3, -6, 98, 106]
    riau_inner = [1,  -0.4, 101, 103.5]
    bboxes = {'Indonesia': indonesia_bbox,
              'Kalimantan': kalimantan,
              'South Sumatra': sumatra_south,
              'Inner Riau': riau_inner}

    land_mask = xr.open_dataset(land_mask)
    ba = pd.read_parquet(ba_prod)
    ba = ba[ba.date.dt.year >= 2008]
    ba15 = ba[ba['date'].dt.year == 2015]

    fwi = xr.open_dataset('~/data/fwi/fwi_dc_indonesia.nc')
    fwi_m = ff['fwi'].groupby('time.month').mean('time') 
    fwi_m_m = fwi_m.where(land_mask)
    #dur = 16
    #dfr.loc[:, 'day_since_tmp'] = dfr['day_since'] * (self.eps / dur)
    ##labs16 = cluster_euc(dfr[['x', 'y', 'z', 'day_since_tmp']].values, self.eps, min_samples=2)
    #dfr.loc[:, 'labs16'] = labs16
    #ba.populate_store_af_blocks(store_name, tropics_store, )
    #ba.cluster_store(store_name, ['Af_tr', 'Am_tr', 'As_tr'])
    #ba.populate_store_tropics(tropics_store)
    #ba.populate_store()
"""
dss = []
for year in range(2008, 2016, 1):
    print(year)
    ffs = glob.glob('/home/tadas/data/cci_ba/{0}/*.*'.format(year))
    for fname in ffs:
        ds = xr.open_dataset(fname)
        print(list(ds.coords))
        ds = spatial_subset(ds['burned_area'], bboxes['Indonesia'])
        dss.append(ds)
        
