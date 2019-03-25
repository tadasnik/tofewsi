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

def dem_ds_to_dfr(cc):
    ds = xr.open_dataset('/mnt/data/dem/era5_dem.nc')
    dem = (ds.z / 9.80665)
    dem = cc.spatial_subset(dem, cc.region_bbox['indonesia'])
    lon_lat_fr = cc.fc[['lonind', 'latind']]
    lon_lat_fr.reset_index(drop = True, inplace = True)
    fr = dem.values[:, lon_lat_fr.latind, lon_lat_fr.lonind]
    dfr = pd.concat([lon_lat_fr[['latind', 'lonind']],
                     pd.DataFrame(fr.T, columns = [str(x) for x in range(1, fr.shape[0] + 1)])],
                     axis = 1)
    return dfr

def prepare_features(cc, current_year):
    first_year = 2001
    cc.read_forest_change()
    cc.frpfr = pd.read_parquet('/mnt/data/frp/frp_count_indonesia_{}deg_monthly_v2.parquet'.format(cc.res))
    cc.frpfr = cc.frpfr[cc.frpfr.frp < 6001]
    frp = pd.merge(cc.fc[['lonind', 'latind']], cc.frpfr, on=['lonind', 'latind'], how='left')
    #fc = pd.merge(frp[['lonind', 'latind']], cc.fc, on=['lonind', 'latind'], how='left')

    frp.fillna(value=0, inplace=True)
    frp = frp.set_index(['lonind', 'latind'])
    frp.drop('frp', axis = 1, inplace = True)
    frp = frp.stack()
    frp = frp.reset_index(name = 'frp')

    #fc = cc.fc.copy()
    #fc = pd.merge(frp[['lonind', 'latind']], cc.fc, on=['lonind', 'latind'], how='left')

    #last year loss
    last_year = cc.fc.drop(['total', 'f_prim', 'gain',
                         '{0}_loss_prim'.format(current_year),
                         '{0}_loss'.format(current_year)], axis = 1)
    last_year_prim = last_year.filter(regex = '^(?=.*prim)', axis = 1)
    last_year_prim = loss_to_features(pd.concat([cc.fc[['lonind', 'latind']],
                     last_year_prim / cc.fc.total.values[:, None]], axis = 1))

    last_year_sec = last_year.filter(regex = '^(?=.*loss)(?!.*prim).*', axis = 1)
    last_year_sec = loss_to_features(pd.concat([cc.fc[['lonind', 'latind']],
                  last_year_sec / cc.fc.total.values[:, None]], axis = 1))


    #lost this year
    this_year = cc.fc.drop(['total', 'f_prim', 'gain',
                         '{0}_loss_prim'.format(first_year),
                         '{0}_loss'.format(first_year)], axis = 1)
    this_year_prim = this_year.filter(regex = '^(?=.*prim)', axis = 1)
    this_year_prim = loss_to_features(pd.concat([cc.fc[['lonind', 'latind']],
                      this_year_prim / cc.fc.total.values[:, None]], axis = 1))

    this_year_sec = this_year.filter(regex = '^(?=.*loss)(?!.*prim).*', axis = 1)
    this_year_sec = loss_to_features(pd.concat([cc.fc[['lonind', 'latind']],
                  this_year_sec / cc.fc.total.values[:, None]], axis = 1))

    #three year loss
    loss_three = cc.fc.drop(['total', 'f_prim', 'gain'], axis = 1)
    three_year_prim = loss_three.filter(regex = '^(?=.*prim)', axis = 1)
    three_year_prim = three_year_prim.rolling(window = 3, min_periods = 2, axis = 1).sum()
    three_year_prim.drop('{0}_loss_prim'.format(first_year), axis = 1, inplace = True)
    three_year_prim = loss_to_features(pd.concat([cc.fc[['lonind', 'latind']],
                      three_year_prim / cc.fc.total.values[:, None]], axis = 1))

    three_year_sec = loss_three.filter(regex = '^(?=.*loss)(?!.*prim).*', axis = 1)
    three_year_sec = three_year_sec.rolling(window = 3, min_periods = 2, axis = 1).sum()
    three_year_sec.drop('{0}_loss'.format(first_year), axis = 1, inplace = True)
    three_year_sec = loss_to_features(pd.concat([cc.fc[['lonind', 'latind']],
                      three_year_sec / cc.fc.total.values[:, None]], axis = 1))

    #accum loss primary
    accum_loss_prim = cc.fc.filter(like = 'loss_prim', axis = 1)
    accum_loss_prim.iloc[: ,:] = accum_loss_prim.iloc[:, :].cumsum(axis = 1)
    accum_loss_prim.drop('{0}_loss_prim'.format(first_year), axis = 1, inplace = True)
    #get prim fraction for each year
    prim_frac = cc.fc.f_prim.values[:, None] - (accum_loss_prim / cc.fc.total.values[:, None])
    prim_frac = loss_to_features(pd.concat([cc.fc[['lonind', 'latind']],
                                                  prim_frac], axis = 1))

    accum_loss_prim = loss_to_features(pd.concat([cc.fc[['lonind', 'latind']],
                      accum_loss_prim / cc.fc.total.values[:, None]], axis = 1))

    #accum loss secondary
    accum_loss_sec = cc.fc.filter(regex = '^(?=.*loss)(?!.*prim).*', axis = 1)
    accum_loss_sec.iloc[: ,:] = accum_loss_sec.iloc[:, :].cumsum(axis = 1)
    accum_loss_sec.drop('{0}_loss'.format(first_year), axis = 1, inplace = True)
    accum_loss_sec = loss_to_features(pd.concat([cc.fc[['lonind', 'latind']],
                      accum_loss_sec / cc.fc.total.values[:, None]], axis = 1))

    #gain
    gain = cc.fc['gain'] / cc.fc['total']
    dem = dem_ds_to_dfr(cc)

    frp.loc[:, 'loss_last_prim'] = last_year_prim.values
    frp.loc[:, 'loss_last_sec'] = last_year_sec.values
    frp.loc[:, 'loss_this_prim'] = this_year_prim.values
    frp.loc[:, 'loss_this_sec'] = this_year_sec.values
    frp.loc[:, 'loss_accum_prim'] = accum_loss_prim.values
    frp.loc[:, 'loss_accum_sec'] = accum_loss_sec.values
    frp.loc[:, 'loss_three_prim'] = three_year_prim.values
    frp.loc[:, 'loss_three_sec'] = three_year_sec.values
    frp.loc[:, 'f_prim'] = prim_frac.values
    frp.loc[:, 'gain'] = gain.repeat(12 * (current_year - first_year)).values
    frp.loc[:, 'dem'] = dem['1'].repeat(12 * (current_year - first_year)).values

    cc.fwi_m = xr.open_dataset(os.path.join(cc.data_path, 'fwi',
                                 'fwi_indonesia_{}deg_monthly_v2.nc'.format(cc.res)))
    rollsum3dc = cc.fwi_m['dc_med'].rolling(time=3,min_periods = 1).sum()
    rollsum3fwi = cc.fwi_m['fwi_med'].rolling(time=3,min_periods = 1).sum()
    cc.fwi_m['dc_3m'] = rollsum3dc
    cc.fwi_m['fwi_3m'] = rollsum3fwi
    cc.fwi_m = cc.fwi_m.sel(time = slice('{0}-01-01'.format(first_year + 1),
                                           '{0}-12-31'.format(current_year)))
    frp = cc.add_fwi_features(list(cc.fwi_m.data_vars.keys()), cc.fc[['lonind', 'latind']], frp)
    frp.rename({'level_2': 'month'}, axis = 1, inplace = True)
    return frp

def prepare_features_5km(cc):
    frp = pd.merge(cc.cc.fc[['lonind', 'latind']], cc.frpfr, on=['lonind', 'latind'], how='inner')
    cc.fc = pd.merge(frp[['lonind', 'latind']], cc.cc.fc, on=['lonind', 'latind'], how='left')

    frp.fillna(value=0, inplace=True)
    frp = frp.set_index(['lonind', 'latind'])
    frp.drop('frp', axis = 1, inplace = True)
    frp = frp.stack()
    frp = frp.reset_index(name = 'frp')

    #cc.fc = cc.cc.fc.copy()
    #cc.fc = pd.merge(frp[['lonind', 'latind']], cc.cc.fc, on=['lonind', 'latind'], how='left')
    part = cc.fc[['f_prim', 'gain']]

    #last year loss
    last_year = cc.fc.drop(['total', 'f_prim', 'gain', 'loss', '17'], axis = 1)
    last_year = loss_to_features(last_year)

    #lost this year
    this_year = cc.fc.drop(['total', 'f_prim', 'gain', 'loss', '1'], axis = 1)
    this_year = loss_to_features(this_year)

    #lost this year
    this_year = fc.drop(['total', 'f_prim', 'gain', 'loss', '1'], axis = 1)
    this_year = loss_to_features(this_year)



    #accum loss
    accum_loss = fc.drop(['total', 'f_prim', 'gain', 'loss'], axis = 1)
    accum_loss.iloc[: ,2:] = accum_loss.iloc[:, 2:].cumsum(axis = 1)
    accum_loss.drop('1', axis = 1, inplace = True)
    accum_loss = loss_to_features(accum_loss)

    #three year loss
    loss_three = fc.drop(['total', 'f_prim', 'gain', 'loss'], axis = 1)
    loss_three.iloc[: ,2:] = loss_three.iloc[:, 2:].rolling(window = 3, min_periods = 2, axis = 1).sum()
    loss_three.drop('1', axis = 1, inplace = True)
    loss_three = loss_to_features(loss_three)


    frp.loc[:, 'loss_last'] = last_year.values
    frp.loc[:, 'loss_this'] = this_year.values
    frp.loc[:, 'loss_three'] = loss_three.values
    frp.loc[:, 'loss_accum'] = accum_loss.values
    frp.loc[:, 'f_prim'] = part['f_prim'].repeat(192).values
    frp.loc[:, 'gain'] = part['gain'].repeat(192).values


    cc.fwi_m = xr.open_dataset(os.path.join(cc.data_path, 'fwi',
                                 'fwi_indonesia_{}deg_monthly_v2.nc'.format(cc.res)))
    rollsum3dc = cc.fwi_m['dc_med'].rolling(time=3,min_periods = 1).sum()
    rollsum3fwi = cc.fwi_m['fwi_med'].rolling(time=3,min_periods = 1).sum()
    cc.fwi_m['dc_3m'] = rollsum3dc
    cc.fwi_m['fwi_3m'] = rollsum3fwi
    cc.fwi_m = cc.fwi_m.sel(time = slice('{0}-01-01'.format(first_year + 1),
                                           '{0}-12-31'.format(current_year)))
    frp = cc.add_fwi_features(list(cc.fwi_m.data_vars.keys()), fc[['lonind', 'latind']], frp)
    frp.rename({'level_2': 'month'}, axis = 1, inplace = True)
    return frp

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
    gr = Gridder(bbox = 'indonesia', step = 0.25)
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

def loss_to_features(fc):
    fc = fc.set_index(['lonind', 'latind'])
    fc = fc.stack()
    fc = fc.repeat(12)
    return fc


class CompData(Envdata):
    def __init__(self, data_path, res, bbox=None, hour=None):
        super().__init__(data_path, bbox=None, hour=None)
        self.res = res
        self.land_mask, self.land_mask_array = self.read_land_mask()

    def read_monthly_dfrs(self, res):
        self.frpfr = pd.read_parquet('/mnt/data/frp/frp_count_indonesia_{}deg_monthly.parquet'.format(res))
        self.dcfr = pd.read_parquet('/mnt/data/fwi/dc_indonesia_{}deg_monthly.parquet'.format(res))
        self.fwifr = pd.read_parquet('/mnt/data/fwi/fwi_indonesia_{}deg_monthly.parquet'.format(res))
        self.ffmcfr = pd.read_parquet('/mnt/data/fwi/ffmc_indonesia_{}deg_monthly.parquet'.format(res))

    def do_piecewise(self, other):
        frp = pd.merge(self.fc[['lonind', 'latind']], self.frpfr, on=['lonind', 'latind'], how='inner')
        comb = pd.merge(frp, other, on=['lonind', 'latind'], how = 'left')
        result = fit_piecewise(comb, 8)
        result[['lonind', 'latind']] = comb[['lonind', 'latind']]
        result.rename({1: 'r2', 0: 'br'}, axis = 1, inplace = True)
        return result

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

    def monthly_fwi_arr_features(self, fwi_arr):
        fwi_m = fwi_arr.resample(time = '1M', closed = 'right').median(dim = 'time')
        fwi_m.rename({'fwi': 'fwi_med', 'ffmc': 'ffmc_med', 'dc': 'dc_med'}, inplace = True)
        fwi_q = fwi_arr.resample(time = '1M', closed = 'right').reduce(np.percentile, q = 75, dim = 'time')
        fwi_q.rename({'fwi': 'fwi_75p', 'ffmc': 'ffmc_75p', 'dc': 'dc_75p'}, inplace = True)
        rolm = fwi_arr.rolling(time=7, min_periods = 1).mean()
        fwi_mm = rolm.resample(time = '1M', closed = 'right').max(dim = 'time')
        fwi_mm.rename({'fwi': 'fwi_7mm', 'ffmc': 'ffmc_7mm', 'dc': 'dc_7mm'}, inplace = True)
        fwi_feat = xr.merge([fwi_m, fwi_q, fwi_mm])

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

    def read_monthly(self, res):
        self.fwi_m = xr.open_dataset(os.path.join(self.data_path, 'fwi',
                                                  'fwi_indonesia_{}deg_monthly_v2.nc'.format(res)))
        self.frp_m = xr.open_dataset(os.path.join(self.data_path, 'frp',
                                                  'frp_count_indonesia_{}deg_monthly_v2.nc'.format(res)))

    def get_pixel(self, lat, lon, fwi_ds):
        frp_pix = self.frp_m['count'].sel(latitude = lat, longitude = lon)
        fwi_pix = self.fwi_m[fwi_ds].sel(latitude = lat, longitude = lon)
        return frp_pix, fwi_pix

    def read_forest_change(self):
        fc = pd.read_parquet('/mnt/data/forest/forest_change_{}deg_v2.parquet'.format(self.res))
        self.fc = fc.fillna(value = 0)

    def digitize_values(self, dfr, columns, bins):
        for column in columns:
            dfr[column + '_d'] = pd.np.digitize(dfr[column], bins = bins)
        return dfr

    def feature_dfr(self, dfr):
        start_year = 2
        last_year_ind = ((dfr.month.astype(int) - 1) // 12) - 1
        last_year_loss = cc.fc

    def fwi_ds_to_dfr(self, prod, lon_lat_fr):
        ds = self.fwi_m[prod]
        lon_lat_fr.reset_index(drop = True, inplace = True)
        fr = ds.values[:, lon_lat_fr.latind, lon_lat_fr.lonind]
        dfr = pd.concat([lon_lat_fr[['latind', 'lonind']],
                         pd.DataFrame(fr.T, columns = [str(x) for x in range(1, fr.shape[0] + 1)])],
                         axis = 1)
        return dfr

    def stack_dfr(self, dfr):
        dfr = dfr.set_index(['lonind', 'latind'])
        dfr = dfr.stack()
        return dfr

    def add_fwi_features(self, products, lon_lat_fr, dfr):
        for prod in products:
            df = self.fwi_ds_to_dfr(prod, lon_lat_fr)
            df = self.stack_dfr(df)
            dfr.loc[:, prod] = df.values
        return dfr

    def read_features(self, res, frp_thresh, total_thresh):
        self.read_monthly_dfrs(res)
        self.read_monthly(res)
        self.read_forest_change(res)
        self.frpfr = self.frpfr[cc.frpfr.frp < frp_thresh]
        self.fc = self.fc[self.fc.total > total_thresh]
        self.fc.reset_index(drop = True, inplace = True)
        self.frpfr.reset_index(drop = True, inplace = True)



if __name__ == '__main__':
    data_path = '/mnt/data/'
    res = 0.25
    #data_path = '/home/tadas/data/'
    cc = CompData(data_path, res)

    #do 5km 
    """
    #Sumatra only!
    cc.frpfr = cc.frpfr[(cc.frpfr.lonind > 43) & (cc.frpfr.lonind < 275)]
    cc.frpfr = cc.frpfr[(cc.frpfr.latind > 141) & (cc.frpfr.latind < 380)]

    cc.frpfr = cc.frpfr[cc.frpfr.frp < 1001]
    cc.fc = cc.fc[cc.fc.total > 35000]
    """

    #do 25 km
    #cc.read_features(0.25, 6001, 700000)

    #res = cc.do_piecewise(cc.ffmcfr)
