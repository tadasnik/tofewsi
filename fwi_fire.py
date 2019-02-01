import os
import datetime
import numpy as np
import xarray as xr
import pandas as pd
import dask.dataframe as dd
import swifter
import pwlf
from envdata import Envdata
from gridding import Gridder
import matplotlib.pyplot as plt

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

def monthly_frp_dfr(frp_file, bbox, res, ds):
    dfr = pd.read_parquet(frp_file)
    gri = Gridder(bbox = 'indonesia', step = res)
    dfr = spatial_subset_dfr(dfr, gri.bbox)
    dfr = gri.add_grid_inds(dfr)
    dfr['year'] = dfr['date'].dt.year
    dfr['month'] = dfr['date'].dt.month
    dfr['mind'] = (dfr['year'] - dfr['year'].min()) * 12 + dfr['month']
    grdfr = pd.DataFrame({'frp': dfr.groupby(['lonind', 'latind', 'mind'])['date'].count()})
    #grdfr = pd.DataFrame({'frp': dfr.groupby(['lonind', 'latind'])['date'].count()})
    grdfr.reset_index(inplace = True)
    grid = np.zeros((gri.lats.shape[0], gri.lons.shape[0], grdfr.mind.max()), dtype=int)
    grid[grdfr.latind, grdfr.lonind, grdfr.mind - 1] = grdfr['frp'].astype(int)
    prim = pd.read_parquet('/mnt/data/forest/forest_primary_{}deg_clean.parquet'.format(res))
    grdfr_agg = pd.DataFrame({'frp': dfr.groupby(['lonind', 'latind'])['date'].count()})
    prim_frp = pd.merge(prim, grdfr_agg, how='inner', on=['lonind', 'latind'])
    frp_m = grid[prim_frp.latind, prim_frp.lonind, :]
    df = pd.concat([prim_frp[['lonind', 'latind', 'frp']], pd.DataFrame(frp_m, columns=[str(x) for x in range(1, 193)])], axis = 1)
    #grid = np.flip(grid, axis = 0)
    #dataset = xr.Dataset({'count': (['latitude', 'longitude', 'time'], grids)},
    #                      coords={'latitude': self.lats,
    #                              'longitude': self.lons,
    #                              'time': dates})

def fit_piecewise(dfr, n_cpu):
    t0 = datetime.datetime.now()
    result = dd.from_pandas(dfr, npartitions = n_cpu).map_partitions(lambda df: df.apply(piecewise,
    axis=1, result_type='expand' )).compute(scheduler='processes')
    print("Dask completed in: {0} using {1} CPUs".format(datetime.datetime.now() - t0, n_cpu))
    return result


def piecewise(row):
    x_vals = row.filter(regex = '_x')
    y_vals = row.filter(regex = '_y')
    row_model = pwlf.PiecewiseLinFit(x_vals, y_vals)
    #fit data for two of segments
    res = row_model.fitfast(2, pop=3)
    r_sq = row_model.r_squared()
    xHat = np.linspace(x_vals.min(), x_vals.max(), num=100)
    yHat = row_model.predict(xHat)
    return res[1], r_sq

def plot_piecewise(dfr, row_ind):
    row = dfr.iloc[row_ind,:]
    x_vals = row.filter(regex = '_x')
    y_vals = row.filter(regex = '_y')
    row_model = pwlf.PiecewiseLinFit(x_vals, y_vals)
    #fit data for two of segments
    res = row_model.fit(2)
    r_sq = row_model.r_squared()
    xHat = np.linspace(x_vals.min(), x_vals.max(), num=100)
    yHat = row_model.predict(xHat)
    plt.plot(x_vals.values, y_vals.values, 'o')
    plt.title('break point {0}, r squeared {1:.2f}'.format(int(res[1]), r_sq))
    plt.plot(xHat, yHat, '-')
    plt.xlabel('DC')
    plt.ylabel('FRP count')
    plt.savefig('figs/PiecewiseLinFit_cell_{0}'.format(row_ind))
    plt.show()

class CompData(Envdata):
    def __init__(self, data_path, bbox=None, hour=None):
        super().__init__(data_path, bbox=None, hour=None)
        self.land_mask, self.land_mask_array = self.read_land_mask()

    def read_monthly_dfrs_25(self):
        self.frpfr25 = pd.read_parquet('/mnt/data/frp/frp_count_indonesia_25km_monthly.parquet')
        self.dcfr25 = pd.read_parquet('/mnt/data/fwi/dc_indonesia_25km_monthly.parquet')
        self.fwifr25 = pd.read_parquet('/mnt/data/fwi/fwi_indonesia_25km_monthly.parquet')


    def read_monthly_dfrs(self):
        self.frpfr = pd.read_parquet('/mnt/data/frp/frp_count_indonesia_5km_monthly.parquet')
        self.dcfr = pd.read_parquet('/mnt/data/fwi/dc_indonesia_5km_monthly.parquet')
        self.fwifr = pd.read_parquet('/mnt/data/fwi/fwi_indonesia_5km_monthly.parquet')

    def do_piecewise(self):
        comb = pd.merge(cc.dcfr, cc.frpfr, on=['lonind', 'latind'])

    def read_lulc(self):
        lulc_path = os.path.join(self.data_path,
                                 'land_cover/peatlands',
                                 'Per-humid_SEA_LULC_0.25.nc')
        self.lulc = xr.open_dataset(lulc_path)

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
        self.fwi_m = xr.open_dataset(os.path.join(self.data_path, 'fwi', 'fwi_indonesia_5km_monthly.nc'))
        self.frp_m = xr.open_dataset(os.path.join(self.data_path, 'frp', 'frp_count_indonesia_5km_monthly.nc'))

    def read_monthly_25(self):
        self.fwi_m25 = xr.open_dataset(os.path.join(self.data_path, 'fwi', 'fwi_arr_m.nc'))
        self.frp_m25 = xr.open_dataset(os.path.join(self.data_path, 'frp', 'frp_count_indonesia_m.nc'))


    def get_pixel(self, lat, lon, fwi_ds):
        frp_pix = self.frp_m['count'].sel(latitude = lat, longitude = lon)
        fwi_pix = self.fwi_m[fwi_ds].sel(latitude = lat, longitude = lon)
        return frp_pix, fwi_pix

    def read_forest_change(self, res):
        self.fc = pd.read_parquet('/mnt/data/forest/forest_change_0.05.parquet')

    def digitize_values(self, dfr, columns, bins):
        for column in columns:
            dfr[column + '_d'] = pd.np.digitize(dfr[column], bins = bins)
        return dfr

if __name__ == '__main__':
    data_path = '/mnt/data/'
    #data_path = '/home/tadas/data/'
    cc = CompData(data_path)
    cc.read_monthly_dfrs()
    #cc.read_monthly_dfrs_25()
    #cc.read_monthly_25()
    cc.read_monthly()
    cc.read_forest_change(0.05)
    cc.frpfr = cc.frpfr[cc.frpfr.frp < 1001]
    cc.fc = cc.fc[cc.fc.total > 35000]
    cc.fc = cc.digitize_values(cc.fc,['gain', 'loss'], [0, 0.01, 0.25, 0.5, 0.75, 1])
    ff = pd.merge(cc.frpfr[['lonind', 'latind', 'frp']], cc.fc, on=['lonind', 'latind'], how='inner')
    med_frp = ff.groupby(['loss_d'])['frp'].median()
    #cc.read_monthly_land_dfr()
