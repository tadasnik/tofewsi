import os
import glob
import datetime
import numpy as np
import xarray as xr
import pandas as pd
import dask.dataframe as dd
import swifter
import pwlf
import geopandas as gpd
from envdata import Envdata
from gridding import Gridder
import matplotlib.pyplot as plt

def add_rolling_features(ds):
    rollsum3dc = ds['dc_med'].rolling(time=3,min_periods = 1).sum()
    rollsum3tp = ds['tp_med'].rolling(time=3,min_periods = 1).sum()
    rollsum3fwi = ds['fwi_med'].rolling(time=3,min_periods = 1).sum()
    ds['dc_3m'] = rollsum3dc
    ds['tp_3m'] = rollsum3tp
    ds['fwi_3m'] = rollsum3fwi
    return ds

def unique_cells(frp):
    fr = frp[(frp.f_prim > 0.8)]
    uni = fr.groupby(['lonind', 'latind'])['fire'].sum().reset_index()
    uni['prim_loss_sum'] = fr.groupby(['lonind', 'latind'])['prim_loss_sum'].mean().values

def repeated_burns(frp):
    #select data
    dfr = frp[(frp.f_prim > 0.8) & (frp.frp > 0)]
    unique_cells = dfr.groupby(['lonind', 'latind', 'year', 'labs8'])
    fbconmin = unique_cells['year'].transform(min)
    fbconmax = unique_cells['year'].transform(max)
    fbr = dfr.query('year == @fbcon')

def repeated_burns_brutal(frp):
    #select data
    dfr = frp[(frp.frp > 0)]
    dfs = []
    for nr, year in enumerate(range(2002, 2019, 1)):
        df = dfr[dfr.year == year]
        pass

    unique_cells = dfr.groupby(['lonind', 'latind', 'year', 'labs8'])
    fbconmin = unique_cells['year'].transform(min)
    fbconmax = unique_cells['year'].transform(max)
    fbr = dfr.query('year == @fbcon')

def add_year_fires(dfr):
    dfr = add_year_fires(cc.frpfr)

def centroids_pandas(self, dfr, dur):
    dates = pd.date_range('2002-01-01', periods = dfr.day_since.max(), freq='d')
    gr = dfr.groupby(['labs8'])['day_since']
    condition_limit = gr.transform(min)

    reduced_dfr = dfr.query('day_since == @condition_limit')
    #centroids = reduced_dfr.groupby(['labs1', 'day_since']).agg({'':'mean', 'latitude':'mean'})
    centroids = reduced_dfr.groupby(['labs1', 'day_since']).agg({'longitude':'mean', 'latitude':'mean'})
    centroids.reset_index(level=1, inplace=True)
    centroids.reset_index(drop=True, inplace=True)
    centroids.loc[:, 'date'] = dates[centroids.day_since-1]
    centroids.loc[:, 'year'] = centroids.date.dt.year
    return centroids

def add_coords_from_ind(dfr, gri):
    dfr['longitude'] = gri.lons[dfr.lonind]
    dfr['latitude'] = gri.lats[dfr.latind]
    return dfr

def dem_ds_to_dfr(cc):
    ds = xr.open_dataset('/mnt/data/dem/era5_dem.nc')
    dem = (ds.z / 9.80665)
    dem = cc.spatial_subset(dem, cc.region_bbox['indonesia'])
    dem = dem.to_dataframe().reset_index()
    gri = Gridder(bbox = 'indonesia', step = res)
    dem = gri.add_grid_inds(dem)
    dem = dem.rename({'z': 'dem'}, axis = 1)
    return dem[['lonind', 'latind', 'dem']]

def deforestation_before_after_all(dfr):
    years = range(2002, 2020, 1)
    dfrs = []
    loss_prim = dfr.filter(like = 'loss_prim', axis = 1)
    loss_sec = dfr.filter(regex = '^(?=.*loss)(?!.*prim).*', axis = 1)
    for nr, year in enumerate(years):
        print(year)
        dfrsel = dfr[dfr.year == year]
        print(dfrsel.shape)
        for sh in range(-3, 4, 1):
            print(sh)
            loss = loss_prim.shift(sh, axis = 1)[dfr.year == year].iloc[:, nr]
            lossec = loss_sec.shift(sh, axis = 1)[dfr.year == year].iloc[:, nr]
            dfrsel[str(-(sh))] = loss
            dfrsel[str(-(sh))+'sec'] = lossec
        dfrs.append(dfrsel)
    dfrall = pd.concat(dfrs)
    dfrall.sort_index(inplace = True)
    return dfrall


def deforestation_before_after(dfr):
    years = range(2002, 2020, 1)
    dfrs = []
    loss_prim = dfr.filter(like = 'loss_prim', axis = 1)
    loss_sec = dfr.filter(regex = '^(?=.*loss)(?!.*prim).*', axis = 1)
    for nr, year in enumerate(years, start = -2):
        dfrsel = dfr[dfr.year == year]
        print(year)
        if (year > 2004) & (year < 2016):
            print(year)
            print(nr)
            loss = loss_prim[dfr.year == year].iloc[:, nr:nr+7]
            loss.columns = [str(x) for x in list(range(-3, 4, 1))]
            lossec = loss_sec[dfr.year == year].iloc[:, nr:nr+7]
            lossec.columns = [str(x) + 'sec' for x in list(range(-3, 4, 1))]
        else:
            losscolumns = [str(x) for x in list(range(-3, 4, 1))]
            loss = pd.DataFrame(0, index=loss_prim[dfr.year == year].index, columns= losscolumns)
            seccolumns = [str(x) + 'sec' for x in list(range(-3, 4, 1))]
            lossec =pd.DataFrame(0, index=loss_prim[dfr.year == year].index, columns = seccolumns)
        dfrsel = pd.concat([dfrsel, loss, lossec], axis = 1)
        dfrs.append(dfrsel)
    dfrall = pd.concat(dfrs)
    dfrall.sort_index(inplace = True)
    return dfrall

def prepare_features_mod_fire_forest_loss(cc, current_year):
    ##USED this
    #first_year = 2001
    #cc.frpfr = pd.read_parquet('data/frps_clust_indonesia_no_volcanoes_0.01deg_inds_v3.parquet')
    cc.frpfr = pd.read_parquet('/mnt/data/frp/M6_indonesia_clustered_no_volcanoes.parquet')
    #cc.frpfr = cc.frpfr[cc.frpfr.confidence >= 30]
    gri = Gridder(bbox = 'indonesia', step = .01)
    cc.frpfr = gri.add_grid_inds(cc.frpfr)
    prim = pd.read_parquet('/mnt/data/forest/forest_primary_0.01deg_v2.parquet')
    prim['f_prim'] = prim['2'] / prim['total']
    prim['f_sec'] = prim['1'] / prim['total']
    loss = pd.read_parquet('/mnt/data/forest/forest_loss_type_0.01deg_v3.parquet')
    gain = pd.read_parquet('/mnt/data/forest/forest_gain_0.01deg.parquet')
    gain = gain.rename({'total': 'gain'}, axis = 1)
    gain = gain.drop('1', axis = 1)
    change = pd.merge(prim[['lonind', 'latind', 'total', 'f_prim', 'f_sec']], loss, on=['lonind', 'latind'], how='left')
    change = pd.merge(change, gain, on=['lonind', 'latind'], how='left')
    change = change.fillna(0)

    peat = pd.read_parquet('data/indonesia_peatlands.parquet')

    cc.frpfr['year'] = cc.frpfr.date.dt.year
    cc.frpfr['month'] = cc.frpfr.date.dt.month
    dmin = cc.frpfr.groupby(['labs8'])['day_since'].transform('min')
    dmax = cc.frpfr.groupby(['labs8'])['day_since'].transform('max')
    cc.frpfr.loc[:, 'duration'] = dmax - dmin
    #frp = cc.frpfr.groupby(['lonind', 'latind', 'year']).size().reset_index(name = 'count')
    frp = cc.frpfr[['lonind', 'latind', 'frp', 'year', 'month', 'labs1', 'labs8', 'day_since', 'duration', 'confidence', 'daynight']]
    frp['fsize'] = frp.groupby('labs8')['labs8'].transform('size')
    frp = pd.merge(change, frp, on=['lonind', 'latind'], how='left')
    frp = pd.merge(frp, peat[['lonind', 'latind', 'peat']], on=['lonind', 'latind'], how='left')
    #fc = pd.merge(frp[['lonind', 'latind']], cc.fc, on=['lonind', 'latind'], how='left')

    frp.fillna(value=0, inplace=True)
    frpd = deforestation_before_after_all(frp[frp.frp > 0])
    frp = pd.concat([frp[frp.frp == 0], frpd])
    frp.fillna(value=0, inplace=True)
    #frp = frp.set_index(['lonind', 'latind'])
    #frp.drop('frp', axis = 1, inplace = True)
    #frp['year'] = frp.date.dt.year
    #frp = frp.stack()
    #frp = frp.reset_index(name = 'frp')

    #f_prim the year before fire
    accum_loss_prim = frp.filter(like = '_loss_prim', axis = 1)
    accum_loss_prim = accum_loss_prim.cumsum(axis = 1)
    years = accum_loss_prim.columns.str.extract('(\d+)').astype(int) + 1
    accum_loss_prim.columns = years[0].tolist()
    accum_loss_prim[0] = 0
    frp['f_prim_before'] = (frp.f_prim.values - (accum_loss_prim.lookup(accum_loss_prim.index, frp.year.values) / frp.total))

    loss_prim_before_fire = accum_loss_prim.lookup(accum_loss_prim.index, (frp.year).astype(int).values)


    #f_sec the year before fire
    accum_loss_sec = frp.filter(regex = '^(?=.*loss)(?!.*prim).*', axis = 1)
    accum_loss_sec = accum_loss_sec.cumsum(axis = 1)
    years = accum_loss_sec.columns.str.extract('(\d+)').astype(int) + 1
    accum_loss_sec.columns = years[0].tolist()
    accum_loss_sec[0] = 0
    frp['f_sec_before'] = (frp.f_sec.values - (accum_loss_sec.lookup(accum_loss_sec.index, frp.year.values) / frp.total))

    loss_sec_before_fire = accum_loss_sec.lookup(accum_loss_sec.index, (frp.year).astype(int).values)


    #accum loss primary
    loss_prim = frp.filter(like = 'loss_prim', axis = 1)
    #loss_prim = loss_prim.rolling(window = 3, min_periods = 1, axis = 1).mean()
    max_loss_prim = loss_prim.idxmax(axis = 1)
    max_loss_prim_year = max_loss_prim.str.extract('(\d+)').astype(int)
    #max_loss_prim_year.reset_index(drop = True, inplace = True)
    frp['max_loss_prim_year'] = max_loss_prim_year[0]
    frp['frp_max_prim_loss'] =  frp.year - frp['max_loss_prim_year']
    frp['prim_loss_sum'] = loss_prim.sum(axis = 1)
    accum_loss_prim = loss_prim.iloc[:, :].cumsum(axis = 1)
    frp['loss_before_max_prim'] = accum_loss_prim.lookup(accum_loss_prim.index, max_loss_prim)

    loss_sec = frp.filter(regex = '^(?=.*loss)(?!.*prim).*', axis = 1)
    #loss_sec = loss_sec.rolling(window = 3, min_periods = 1, axis = 1).mean()
    max_loss_sec = loss_sec.idxmax(axis = 1)
    max_loss_sec_year = max_loss_sec.str.extract('(\d+)').astype(int)
    #max_loss_sec_year.reset_index(drop = True, inplace = True)
    frp['max_loss_sec_year'] = max_loss_sec_year[0]
    frp['frp_max_sec_loss'] = frp.year - frp['max_loss_sec_year']
    frp['sec_loss_sum'] = loss_sec.sum(axis = 1)

    accum_loss_sec = iloc = loss_sec.iloc[:, :].cumsum(axis = 1)
    frp['loss_before_max_sec'] = accum_loss_sec.lookup(accum_loss_sec.index, max_loss_sec)
    frp['loss_after_max_sec'] = frp['sec_loss_sum'] - frp['loss_before_max_sec']

    frp['loss_prim_before_fire'] = loss_prim_before_fire
    frp['loss_sec_before_fire'] = loss_sec_before_fire

    frp['loss_sec_after_fire'] = frp['sec_loss_sum'] - frp['loss_sec_before_fire']
    frp['loss_prim_after_fire'] = frp['prim_loss_sum'] - frp['loss_prim_before_fire']


    frp['loss_total'] = frp['prim_loss_sum'] + frp['sec_loss_sum']
    #from random import randint
    #frp['randoms_prim'] = frp['max_loss_prim_year'] - [randint(2001, 2019) for x in range(1, frp.shape[0] + 1)]
    #frp['randoms_sec'] = frp['max_loss_sec_year'] - [randint(2001, 2019) for x in range(1, frp.shape[0] + 1)]
    frp['fire'] = 0
    frp.loc[:, 'fire'][frp.frp > 0] = 1
    frp['peat'][frp.peat > 0] = 1
    frp['prim_loss_before_max'] = frp.groupby('labs8')['loss_prim_before_fire'].transform('max')
    frp['f_prim_any'] = frp.groupby('labs8')['f_prim'].transform('max')
    frp['peat_any'] = frp.groupby('labs8')['peat'].transform('sum')
    frp['peat_any'][frp.peat_any > 0] = 1
    frp = add_coords_from_ind(frp, gri)
    frp['daynight'] = frp.daynight.astype(str)
    #STOPS HERE!!!



    #fc = cc.fc.copy()
    #fc = pd.merge(frp[['lonind', 'latind']], cc.fc, on=['lonind', 'latind'], how='left')

    #peak deforestation year
    last_year = cc.fc.drop(['total', 'f_prim',
                         '{0}_loss_prim'.format(current_year),
                         '{0}_loss'.format(current_year)], axis = 1)
    last_year_prim = last_year.filter(regex = '^(?=.*prim)', axis = 1)
    last_year_prim = loss_to_features(pd.concat([cc.fc[['lonind', 'latind']],
                     last_year_prim / cc.fc.total.values[:, None]], axis = 1))

    last_year_sec = last_year.filter(regex = '^(?=.*loss)(?!.*prim).*', axis = 1)
    last_year_sec = loss_to_features(pd.concat([cc.fc[['lonind', 'latind']],
                  last_year_sec / cc.fc.total.values[:, None]], axis = 1))



    #last year loss
    last_year = cc.fc.drop(['total', 'f_prim',
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
    loss_prim = frp.filter(like = 'loss_prim', axis = 1)
    max_loss_prim = loss_prim.idxmax(axis = 1)
    max_loss_prim_year = max_loss_prim.str.extract('(\d+)').astype(int)
    #max_loss_prim_year.reset_index(drop = True, inplace = True)
    frp['max_loss_prim_year'] = max_loss_prim_year[0]
    frp['frp_max_prim_loss'] = frp['max_loss_prim_year'] - frp.date.dt.year
    frp['prim_loss_sum'] = loss_prim.sum(axis = 1)

    loss_sec = frp.filter(regex = '^(?=.*loss)(?!.*prim).*', axis = 1)
    max_loss_sec = loss_sec.idxmax(axis = 1)
    max_loss_sec_year = max_loss_sec.str.extract('(\d+)').astype(int)
    #max_loss_sec_year.reset_index(drop = True, inplace = True)
    frp['max_loss_sec_year'] = max_loss_sec_year[0]
    frp['frp_max_sec_loss'] = frp['max_loss_sec_year'] - frp.date.dt.year
    frp['sec_loss_sum'] = loss_sec.sum(axis = 1)

    accum_loss_sec.iloc[: ,:] = accum_loss_sec.iloc[:, :].cumsum(axis = 1)
    accum_loss_sec.drop('{0}_loss'.format(first_year), axis = 1, inplace = True)
    accum_loss_sec.reset_index(drop = True, inplace = True)
    prim_frac = frp.f_prim.values[:, None] - (accum_loss_prim / frp.total.values[:, None])

    accum_loss_prim.iloc[: ,:] = loss_prim.iloc[:, :].cumsum(axis = 1)
    accum_loss_prim.drop('{0}_loss_prim'.format(first_year), axis = 1, inplace = True)
    accum_loss_prim.reset_index(drop = True, inplace = True)
    prim_frac = frp.f_prim.values[:, None] - (accum_loss_prim / frp.total.values[:, None])
    indss = frp.year.astype(str) + '_loss_prim'
    prim_loss_frp = prim_frac.lookup(accum_loss_prim.index, indss)

    frp['prim_loss_frp'] = prim_loss_frp

    #accum loss secondary
    accum_loss_sec = frp.filter(regex = '^(?=.*loss)(?!.*prim).*', axis = 1)
    accum_loss_sec.iloc[: ,:] = accum_loss_sec.iloc[:, :].cumsum(axis = 1)
    accum_loss_sec.drop('{0}_loss'.format(first_year), axis = 1, inplace = True)
    accum_loss_sec.reset_index(drop = True, inplace = True)
    prim_frac = frp.f_prim.values[:, None] - (accum_loss_prim / frp.total.values[:, None])
    indss = frp.year.astype(str) + '_loss_prim'




    #get prim fraction for each year
    prim_frac = frp.f_prim.values[:, None] - (accum_loss_prim / frp.total.values[:, None])
    prim_frac = loss_to_features(pd.concat([cc.fc[['lonind', 'latind']],
                                                  prim_frac], axis = 1))

    accum_loss_prim = loss_to_features(pd.concat([cc.fc[['lonind', 'latind']],
                      accum_loss_prim / cc.fc.total.values[:, None]], axis = 1))

    accum_loss_sec = frp.filter(regex = '^(?=.*loss)(?!.*prim).*', axis = 1)
    accum_loss_sec.iloc[: ,:] = accum_loss_sec.iloc[:, :].cumsum(axis = 1)
    accum_loss_sec.drop('{0}_loss'.format(first_year), axis = 1, inplace = True)
    accum_loss_sec = loss_to_features(pd.concat([cc.fc[['lonind', 'latind']],
                      accum_loss_sec / cc.fc.total.values[:, None]], axis = 1))

    #gain
    gain = cc.fc['gain'] / cc.fc['total']
    dem = dem_ds_to_dfr(cc)

    #peat depth
    depth = peat_depth_dfr(cc)

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
    frp = pd.merge(frp, dem, on = ['lonind', 'latind'], how = 'left')
    frp = pd.merge(frp, depth, on = ['lonind', 'latind'], how = 'left')

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

def prepare_features_fire_forest_loss(cc, current_year):
    current_year = 2018
    first_year = 2001
    #cc.frpfr = pd.read_parquet('data/frps_clust_indonesia_no_volcanoes_0.01deg_inds_v3.parquet')
    cc.frpfr = pd.read_parquet('/mnt/data/frp/M6_indonesia_clustered.parquet')
    cc.frpfr['year'] = cc.frpfr.date.dt.year
    #cc.frpfr = cc.frpfr[cc.frpfr.year < 2019]
    #cc.frpfr = cc.frpfr[cc.frpfr.confidence >= 30]
    gri = Gridder(bbox = 'indonesia', step = .01)
    cc.frpfr = gri.add_grid_inds(cc.frpfr)
    prim = pd.read_parquet('/mnt/data/forest/forest_primary_0.01deg_v2.parquet')
    prim['f_prim'] = prim['2'] / prim['total']
    prim['f_sec'] = prim['1'] / prim['total']
    loss = pd.read_parquet('/mnt/data/forest/forest_loss_type_0.01deg_v3.parquet')
    gain = pd.read_parquet('/mnt/data/forest/forest_gain_0.01deg.parquet')
    gain = gain.rename({'total': 'gain'}, axis = 1)
    gain = gain.drop('1', axis = 1)
    change = pd.merge(prim[['lonind', 'latind', 'total', 'f_prim', 'f_sec']], loss, on=['lonind', 'latind'], how='left')
    change = pd.merge(change, gain, on=['lonind', 'latind'], how='left')
    peat = pd.read_parquet('data/indonesia_peatlands.parquet')

    dmin = cc.frpfr.groupby(['labs8'])['day_since'].transform('min')
    dmax = cc.frpfr.groupby(['labs8'])['day_since'].transform('max')
    cc.frpfr.loc[:, 'duration'] = dmax - dmin
    #frp = cc.frpfr.groupby(['lonind', 'latind', 'year']).size().reset_index(name = 'count')
    frp = cc.frpfr[['lonind', 'latind', 'frp', 'date', 'year', 'labs1', 'labs8', 'day_since', 'duration', 'confidence', 'daynight']]
    frp['fsize'] = frp.groupby('labs8')['labs8'].transform('size')
    frp = pd.merge(change, frp, on=['lonind', 'latind'], how='left')
    frp = pd.merge(frp, peat[['lonind', 'latind', 'peat']], on=['lonind', 'latind'], how='left')
    #fc = pd.merge(frp[['lonind', 'latind']], cc.fc, on=['lonind', 'latind'], how='left')

    frp.fillna(value=0, inplace=True)
    frpd = deforestation_before_after(frp[frp.frp > 0])
    frp = pd.concat([frp[frp.frp == 0], frpd])
    frp.fillna(value=0, inplace=True)
    #frp = frp.set_index(['lonind', 'latind'])
    #frp.drop('frp', axis = 1, inplace = True)
    #frp['year'] = frp.date.dt.year
    #frp = frp.stack()
    #frp = frp.reset_index(name = 'frp')

    #accum loss primary
    loss_prim = cc.fc.filter(like = 'loss_prim', axis = 1)
    #loss_prim = loss_prim.rolling(window = 3, min_periods = 1, axis = 1).mean()
    max_loss_prim = loss_prim.idxmax(axis = 1)
    max_loss_prim_year = max_loss_prim.str.extract('(\d+)').astype(int)
    #max_loss_prim_year.reset_index(drop = True, inplace = True)
    frp['max_loss_prim_year'] = max_loss_prim_year[0]
    frp['frp_max_prim_loss'] =  frp.year - frp['max_loss_prim_year']
    frp['prim_loss_sum'] = loss_prim.sum(axis = 1)

    loss_sec = cc.fc.filter(regex = '^(?=.*loss)(?!.*prim).*', axis = 1)
    max_loss_sec = loss_sec.idxmax(axis = 1)
    max_loss_sec_year = max_loss_sec.str.extract('(\d+)').astype(int)
    #max_loss_sec_year.reset_index(drop = True, inplace = True)
    frp['max_loss_sec_year'] = max_loss_sec_year[0]
    frp['frp_max_sec_loss'] = frp['max_loss_sec_year'] - frp.date.dt.year
    frp['sec_loss_sum'] = loss_sec.sum(axis = 1)


    accum_loss_prim = loss_prim.iloc[:, :].cumsum(axis = 1)
    frp['loss_before_max_prim'] = accum_loss_prim.lookup(accum_loss_prim.index, max_loss_prim)


    #frp['loss_prim_before_fire'] = 0
    #frpsel = frp[frp.year > 0]
    #acc_sel = accum_loss_prim[frp.year > 0]
    #acc_sel.columns = list(range(2001, 2019, 1))
    #loss_prim_before_fire = acc_sel.lookup(acc_sel.index, (frpsel.year).astype(int).values)
    #frp['loss_prim_before_fire'][frp.year > 0] = loss_prim_before_fire
    #frp['loss_prim_after_fire'] = frp['prim_loss_sum'] - frp['loss_prim_before_fire']

    frp['loss_prim_before_fire'] = 0
    accum_loss_prim.columns = list(range(2001, 2019, 1))
    accum_loss_prim[-1] = 0
    loss_prim_before_fire = accum_loss_prim.lookup(accum_loss_prim.index, (frp.year - 1).astype(int).values)
    frp['loss_prim_before_fire'] = loss_prim_before_fire
    frp['loss_prim_after_fire'] = frp['prim_loss_sum'] - frp['loss_prim_before_fire']

    loss_sec = frp.filter(regex = '^(?=.*loss)(?!.*prim).*', axis = 1)
    #loss_sec = loss_sec.rolling(window = 3, min_periods = 1, axis = 1).mean()
    max_loss_sec = loss_sec.idxmax(axis = 1)
    max_loss_sec_year = max_loss_sec.str.extract('(\d+)').astype(int)
    #max_loss_sec_year.reset_index(drop = True, inplace = True)
    frp['max_loss_sec_year'] = max_loss_sec_year[0]
    frp['frp_max_sec_loss'] = frp.year - frp['max_loss_sec_year']
    frp['sec_loss_sum'] = loss_sec.sum(axis = 1)

    accum_loss_sec = iloc = loss_sec.iloc[:, :].cumsum(axis = 1)
    frp['loss_before_max_sec'] = accum_loss_sec.lookup(accum_loss_sec.index, max_loss_sec)
    frp['loss_after_max_sec'] = frp['sec_loss_sum'] - frp['loss_before_max_sec']

    frp['loss_sec_before_fire'] = 0
    accum_loss_sec.columns = list(range(2001, 2019, 1))
    accum_loss_sec[-1] = 0
    accum_loss_sec[2019] = 0
    loss_sec_before_fire = accum_loss_sec.lookup(accum_loss_sec.index, (frp.year + 1).astype(int).values)
    frp['loss_sec_before_fire'] = loss_sec_before_fire
    frp['loss_sec_after_fire'] = frp['sec_loss_sum'] - frp['loss_sec_before_fire']


    frp['loss_total'] = frp['prim_loss_sum'] + frp['sec_loss_sum']
    #from random import randint
    #frp['randoms_prim'] = frp['max_loss_prim_year'] - [randint(2001, 2019) for x in range(1, frp.shape[0] + 1)]
    #frp['randoms_sec'] = frp['max_loss_sec_year'] - [randint(2001, 2019) for x in range(1, frp.shape[0] + 1)]
    frp['fire'] = 0
    frp.loc[:, 'fire'][frp.frp > 0] = 1
    frp['peat'][frp.peat > 0] = 1
    frp['prim_loss_before_max'] = frp.groupby('labs8')['loss_prim_before_fire'].transform('max')
    frp['f_prim_any'] = frp.groupby('labs8')['f_prim'].transform('max')
    frp['peat_any'] = frp.groupby('labs8')['peat'].transform('sum')
    frp['peat_any'][frp.peat_any > 0] = 1
    frp = add_coords_from_ind(frp, gri)


    #fc = cc.fc.copy()
    #fc = pd.merge(frp[['lonind', 'latind']], cc.fc, on=['lonind', 'latind'], how='left')

    #peak deforestation year
    last_year = cc.fc.drop(['total', 'f_prim',
                         '{0}_loss_prim'.format(current_year),
                         '{0}_loss'.format(current_year)], axis = 1)
    last_year_prim = last_year.filter(regex = '^(?=.*prim)', axis = 1)
    last_year_prim = loss_to_features(pd.concat([cc.fc[['lonind', 'latind']],
                     last_year_prim / cc.fc.total.values[:, None]], axis = 1))

    last_year_sec = last_year.filter(regex = '^(?=.*loss)(?!.*prim).*', axis = 1)
    last_year_sec = loss_to_features(pd.concat([cc.fc[['lonind', 'latind']],
                  last_year_sec / cc.fc.total.values[:, None]], axis = 1))



    #last year loss
    last_year = cc.fc.drop(['total', 'f_prim',
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


    accum_loss_sec.iloc[: ,:] = accum_loss_sec.iloc[:, :].cumsum(axis = 1)
    accum_loss_sec.drop('{0}_loss'.format(first_year), axis = 1, inplace = True)
    accum_loss_sec.reset_index(drop = True, inplace = True)
    prim_frac = frp.f_prim.values[:, None] - (accum_loss_prim / frp.total.values[:, None])

    accum_loss_prim.iloc[: ,:] = loss_prim.iloc[:, :].cumsum(axis = 1)
    accum_loss_prim.drop('{0}_loss_prim'.format(first_year), axis = 1, inplace = True)
    accum_loss_prim.reset_index(drop = True, inplace = True)
    prim_frac = frp.f_prim.values[:, None] - (accum_loss_prim / frp.total.values[:, None])
    indss = frp.year.astype(str) + '_loss_prim'
    prim_loss_frp = prim_frac.lookup(accum_loss_prim.index, indss)
    frp['prim_loss_frp'] = prim_loss_frp

    #accum loss secondary
    accum_loss_sec = frp.filter(regex = '^(?=.*loss)(?!.*prim).*', axis = 1)
    accum_loss_sec.iloc[: ,:] = accum_loss_sec.iloc[:, :].cumsum(axis = 1)
    accum_loss_sec.drop('{0}_loss'.format(first_year), axis = 1, inplace = True)
    accum_loss_sec.reset_index(drop = True, inplace = True)
    prim_frac = frp.f_prim.values[:, None] - (accum_loss_prim / frp.total.values[:, None])
    indss = frp.year.astype(str) + '_loss_prim'




    #get prim fraction for each year
    prim_frac = frp.f_prim.values[:, None] - (accum_loss_prim / frp.total.values[:, None])
    prim_frac = loss_to_features(pd.concat([cc.fc[['lonind', 'latind']],
                                                  prim_frac], axis = 1))

    accum_loss_prim = loss_to_features(pd.concat([cc.fc[['lonind', 'latind']],
                      accum_loss_prim / cc.fc.total.values[:, None]], axis = 1))

    accum_loss_sec = frp.filter(regex = '^(?=.*loss)(?!.*prim).*', axis = 1)
    accum_loss_sec.iloc[: ,:] = accum_loss_sec.iloc[:, :].cumsum(axis = 1)
    accum_loss_sec.drop('{0}_loss'.format(first_year), axis = 1, inplace = True)
    accum_loss_sec = loss_to_features(pd.concat([cc.fc[['lonind', 'latind']],
                      accum_loss_sec / cc.fc.total.values[:, None]], axis = 1))

    #gain
    gain = cc.fc['gain'] / cc.fc['total']
    dem = dem_ds_to_dfr(cc)

    #peat depth
    depth = peat_depth_ds_to_dfr(cc)

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
    frp.loc[:, 'peat_depth'] = depth['1'].repeat(12 * (current_year - first_year)).values

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
        rolm = fwi_arr.rolling(time = 7, min_periods = 1).mean()
        fwi_mm = rolm.resample(time = '1M', closed = 'right').max(dim = 'time')
        fwi_mm.rename({'fwi': 'fwi_7mm', 'ffmc': 'ffmc_7mm', 'dc': 'dc_7mm'}, inplace = True)
        fwi_feat = xr.merge([fwi_m, fwi_q, fwi_mm])
        return fwi_feat

    def compute_monthly(self, ds):
        ds_monthly = ds.resample(time = '1M', closed = 'right').mean(dim = 'time')
        return ds_monthly

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
        fc = pd.read_parquet('/mnt/data/forest/forest_change_{}deg_v3.parquet'.format(self.res))
        fc = fc[fc.total > (fc.total.max() * 0.8)]
        self.fc = fc.fillna(value = 0)

    def digitize_values(self, dfr, columns, bins):
        for column in columns:
            dfr[column + '_d'] = pd.np.digitize(dfr[column], bins = bins)
        return dfr

    def feature_dfr(self, dfr):
        start_year = 2
        last_year_ind = ((dfr.month.astype(int) - 1) // 12) - 1
        last_year_loss = cc.fc

    def fwi_ds_to_dfr(self, prod, fwi, lon_lat_fr):
        ds = fwi[prod]
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

    def add_fwi_features(self, products, fwi, lon_lat_fr, dfr):
        for prod in products:
            print(prod)
            df = self.fwi_ds_to_dfr(prod, fwi, lon_lat_fr)
            print(df.shape)
            df = self.stack_dfr(df)
            print(df.shape)
            print(dfr.shape)
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



    def frp_lc_features(self, out_name):
        first_year = 2001
        self.frpfr = pd.read_parquet('/mnt/data/frp/monthly_frp_count_indonesia_0.25deg_2019_10.parquet'.format(self.res, out_name))
        grdfr = pd.read_parquet('/mnt/data/frp/monthly_fire_duration_size_daynight_0.25deg_2019_10.parquet'.format(self.res, out_name))
        grdfr = grdfr.fillna(0)
        current_year = grdfr.year.max()
        self.read_forest_change()
        self.fc = self.fc[self.fc.total > 800000]
        self.fc['{}_loss'.format(current_year)] = 0
        self.fc['{}_loss_prim'.format(current_year)] = 0
        months_to_fill = 12 - grdfr[grdfr.year == current_year].month.max()
        dummy_year = np.zeros_like(self.frpfr.iloc[:, :months_to_fill])
        last_month = int(self.frpfr.columns[-1]) + 1
        columns = [str(x) for x in range(last_month, last_month + months_to_fill, 1)]
        frpfr_plus = pd.concat([self.frpfr, pd.DataFrame(data = dummy_year, columns = columns)], axis = 1)
        #self.frpfr = filter_volcanoes(self.frpfr)
        #self.frpfr = self.frpfr[self.frpfr.frp < 6001]
        frp = pd.merge(self.fc[['lonind', 'latind']], frpfr_plus, on=['lonind', 'latind'], how = 'left')
        #fc = pd.merge(frp[['lonind', 'latind']], self.fc, on=['lonind', 'latind'], how='left')

        frp.fillna(value=0, inplace=True)
        frp = frp.set_index(['lonind', 'latind'])
        frp.drop('frp', axis = 1, inplace = True)

        #accum fire counts
        frp_acc = frp.cumsum(axis = 1)
        frp_acc = frp_acc.shift(1, axis = 1, fill_value = 0)

        frp_acc = frp_acc.stack()
        frp_acc = frp_acc.reset_index(name = 'frp_acc')
        frp = frp.stack()
        frp = frp.reset_index(name = 'frp')
        frp.rename({'level_2': 'mind'}, axis = 1, inplace = True)
        frp['mind'] = frp['mind'].astype(int)
        grdfr = grdfr.drop(['frp'], axis = 1)
        frp = pd.merge(frp, grdfr, on=['lonind', 'latind', 'mind'], how='left')
        frp.fillna(value=0, inplace=True)

        last_year_prim = self.fc.filter(regex = '^(?!{0})(.*loss_prim)'.format(current_year), axis = 1)
        last_year_prim = loss_to_features(pd.concat([self.fc[['lonind', 'latind']],
                         last_year_prim / self.fc.total.values[:, None]], axis = 1))

        last_year_sec = self.fc.filter(regex = '^(?!{0})(.*loss$)'.format(current_year), axis = 1)
        last_year_sec = loss_to_features(pd.concat([self.fc[['lonind', 'latind']],
                      last_year_sec / self.fc.total.values[:, None]], axis = 1))

        #lost this year
        this_year_prim = self.fc.filter(regex = '^(?!{0})(.*loss_prim)'.format(first_year), axis = 1)
        this_year_prim = loss_to_features(pd.concat([self.fc[['lonind', 'latind']],
                          this_year_prim / self.fc.total.values[:, None]], axis = 1))

        this_year_sec = self.fc.filter(regex = '^(?!{0})(.*loss$)'.format(first_year), axis = 1)
        this_year_sec = loss_to_features(pd.concat([self.fc[['lonind', 'latind']],
                      this_year_sec / self.fc.total.values[:, None]], axis = 1))

        #three year loss
        loss_three = self.fc.drop(['total', 'f_prim', 'gain'], axis = 1)
        three_year_prim = loss_three.filter(regex = '^(?=.*prim)', axis = 1)
        three_year_prim = three_year_prim.rolling(window = 3, min_periods = 1, axis = 1).sum()
        three_year_prim.drop('{0}_loss_prim'.format(current_year), axis = 1, inplace = True)
        three_year_prim = loss_to_features(pd.concat([self.fc[['lonind', 'latind']],
                          three_year_prim / self.fc.total.values[:, None]], axis = 1))

        three_year_sec = loss_three.filter(regex = '^(?=.*loss)(?!.*prim).*', axis = 1)
        three_year_sec = three_year_sec.rolling(window = 3, min_periods = 1, axis = 1).sum()
        three_year_sec.drop('{0}_loss'.format(current_year), axis = 1, inplace = True)
        three_year_sec = loss_to_features(pd.concat([self.fc[['lonind', 'latind']],
                          three_year_sec / self.fc.total.values[:, None]], axis = 1))

        #accum loss primary excluding this
        accum_loss_prim = self.fc.filter(like = 'loss_prim', axis = 1)
        accum_loss_prim.drop('{0}_loss_prim'.format(current_year), axis = 1, inplace = True)
        accum_loss_prim = accum_loss_prim.cumsum(axis = 1)
        #accum_loss_prim = accum_loss_prim.shift(1, axis = 1, fill_value = 0)
        ac_loss_prim = loss_to_features(pd.concat([self.fc[['lonind', 'latind']],
                          accum_loss_prim / self.fc.total.values[:, None]], axis = 1))

        #accum loss secondary excluding this
        accum_loss_sec = self.fc.filter(regex = '^(?=.*loss)(?!.*prim).*', axis = 1)
        accum_loss_sec.drop('{0}_loss'.format(current_year), axis = 1, inplace = True)
        accum_loss_sec = accum_loss_sec.cumsum(axis = 1)
        #accum_loss_sec = accum_loss_sec.shift(1, axis = 1, fill_value = 0)
        ac_loss_sec = loss_to_features(pd.concat([self.fc[['lonind', 'latind']],
                          accum_loss_sec / self.fc.total.values[:, None]], axis = 1))

        #accum loss primary
        accum_loss_prim = self.fc.filter(like = 'loss_prim', axis = 1)
        accum_loss_prim.iloc[: ,:] = accum_loss_prim.iloc[:, :].cumsum(axis = 1)
        accum_loss_prim.drop('{0}_loss_prim'.format(first_year), axis = 1, inplace = True)

        #get prim fraction for each year
        prim_frac = self.fc.f_prim.values[:, None] - (accum_loss_prim / self.fc.total.values[:, None])
        prim_frac = loss_to_features(pd.concat([self.fc[['lonind', 'latind']],
                                                      prim_frac], axis = 1))

        accum_loss_prim = loss_to_features(pd.concat([self.fc[['lonind', 'latind']],
                          accum_loss_prim / self.fc.total.values[:, None]], axis = 1))

        #accum loss secondary
        accum_loss_sec = self.fc.filter(regex = '^(?=.*loss)(?!.*prim).*', axis = 1)
        accum_loss_sec.iloc[: ,:] = accum_loss_sec.iloc[:, :].cumsum(axis = 1)
        accum_loss_sec.drop('{0}_loss'.format(first_year), axis = 1, inplace = True)
        accum_loss_sec = loss_to_features(pd.concat([self.fc[['lonind', 'latind']],
                          accum_loss_sec / self.fc.total.values[:, None]], axis = 1))

        #gain
        gain = self.fc['gain'] / self.fc['total']
        dem = dem_ds_to_dfr(self)

        #peat depth
        depth = self.peat_depth_dfr()

        frp.loc[:, 'loss_last_prim'] = last_year_prim.values
        frp.loc[:, 'loss_last_sec'] = last_year_sec.values
        frp.loc[:, 'loss_this_prim'] = this_year_prim.values
        frp.loc[:, 'loss_this_sec'] = this_year_sec.values
        frp.loc[:, 'loss_accum_prim'] = accum_loss_prim.values
        frp.loc[:, 'loss_prim_before'] = ac_loss_prim.values
        frp.loc[:, 'loss_accum_sec'] = accum_loss_sec.values
        frp.loc[:, 'loss_sec_before'] = ac_loss_sec.values
        frp.loc[:, 'loss_three_prim'] = three_year_prim.values
        frp.loc[:, 'loss_three_sec'] = three_year_sec.values
        frp.loc[:, 'frp_acc'] = frp_acc['frp_acc'].values
        frp.loc[:, 'f_prim'] = prim_frac.values
        frp.loc[:, 'gain'] = gain.repeat(12 * (current_year - first_year)).values
        frp = pd.merge(frp, dem, on = ['lonind', 'latind'], how = 'left')
        frp = pd.merge(frp, depth, on = ['lonind', 'latind'], how = 'left')
        frp = frp.fillna(0)
        frp.to_parquet('/mnt/data2/SEAS5/forecast/frp_features_{0}.parquet'.format(out_name))

    def peat_depth_dfr(self):
        pdfr = pd.read_parquet('/mnt/data/land_cover/peat_depth/peat_mask_0.01deg.parquet')
        gri = Gridder(bbox = 'indonesia', step = self.res)
        pdfr = gri.add_grid_inds(pdfr)
        pdfr = pdfr.groupby(['lonind', 'latind'])['peatd'].sum()
        return pdfr

    def add_rolling_features(self, ds):
        rollsum3dc = ds['dc_med'].rolling(time=3,min_periods = 1).sum()
        rollsum3tp = ds['tp_med'].rolling(time=3,min_periods = 1).sum()
        rollsum3fwi = ds['fwi_med'].rolling(time=3,min_periods = 1).sum()
        ds['dc_3m'] = rollsum3dc
        ds['tp_3m'] = rollsum3tp
        ds['fwi_3m'] = rollsum3fwi
        return ds

if __name__ == '__main__':
    data_path = '/mnt/data/'
    res = 0.01
    out_name = '2019_12'
    #res = 0.01
    #data_path = '/home/tadas/data/'
    cc = CompData(data_path, res)
    current_year = 2019
    first_year = 2001
    cc.read_forest_change()

    #cc.frp_lc_features(out_name)

