import glob
import pandas as pd
from envdata import Envdata
from gridding import Gridder

def read_atsr():
    fnames = glob.glob('/home/tadas/forests_fire/atsr/*.FIRE')
    dss = []
    for fname in fnames:
        df = pd.read_csv(fname, header = None, delim_whitespace = True,
                names = ['date', 'orb', 'time', 'latitude', 'longitude'])
        dss.append(df)
    dfr = pd.concat(dss)
    return dfr

def spatial_subset_dfr(dfr, bbox):
    dfr = dfr[(dfr['latitude'] < bbox[0]) &(dfr['latitude'] > bbox[2])]
    dfr = dfr[(dfr['longitude'] > bbox[1]) &(dfr['longitude'] < bbox[3])]
    return dfr

def get_unique(frp):
    uni = frp.groupby(['lonind', 'latind', 'year'])['fire'].sum().unstack(fill_value = 0)
    """
    unirest = frp.groupby(['lonind', 'latind']).agg({'prim_loss_sum': 'first',
                                                 'f_prim': 'first',
                                                 'f_sec': 'first',
                                                 'year': 'nunique',
                                                 'labs8': 'first',
                                                 'lon': 'first',
                                                 'lat': 'first',
                                                 'duration': 'mean',
                                                 'peat': 'first',
                                                 'gain': 'first',
                                                 'fsize': 'mean',
                                                 'loss_prim_before_fire': 'mean',
                                                 'total': 'first',
                                                 'latind': 'first',
                                                 'confidence': 'mean'})
    return pd.concat([uni, unirest], axis = 1)
    """
    return uni

gri = Gridder(bbox = 'indonesia', step = 0.01)
#data_dir = '/home/tadas/tofewsi/data'
atsr = pd.read_parquet('/home/tadas/pyviz/data/atsr_indo_1997_2002.parquet')
atsr = spatial_subset_dfr(atsr, [5.8, 95, -5.935, 119])
atsr = atsr[~((atsr.longitude > 118) & (atsr.latitude < -2.5))]
atsr = gri.add_grid_inds(atsr)
atsr['year'] = atsr.date.dt.year
atsr['fire'] = 1


#en = Envdata(data_dir)
#frp = pd.read_parquet('/home/tadas/tofewsi/data/frp_grouped_peat_daynight.parquet')
frp = pd.read_parquet('/home/tadas/tofewsi/data/frp_grouped_daynight_2019_11.parquet')
#sumatra+kali
frp = spatial_subset_dfr(frp, [5.8, 95, -5.935, 119])
frp = frp[~((frp.longitude > 118) & (frp.latitude < -2.5))]
frpn = frp[frp.daynight == 'N']
frp = frp[(frp.duration > 1)]# & (frp.labs8.isin(frpn.labs8))]
frpn = frp.groupby(['labs8'])['daynight'].value_counts().unstack(fill_value = 0)
columns = ['lonind', 'latind', 'longitude', 'latitude', 'year', 'fire', 'f_prim_before', 'duration']
atsr = atsr[atsr.year < 2002]
atsr['f_prim_before'] = 0
atsr['duration'] = 0
comb = pd.concat([atsr[columns], frp[frp.fire > 0][columns]])
comb = get_unique(comb)
comb.columns = comb.columns.astype(int)

test = comb.copy()
test[comb > 0] = 1
combcs = test.cumsum(axis = 1)
combcs = combcs.reset_index()
combcs = combcs.melt(id_vars = ['lonind', 'latind'], var_name = 'year', value_name = 'repeat')
#combcs.to_parquet('~/pyviz/data/repeated_2019_11.parquet')
frpr = pd.merge(frp[frp.fire > 0], combcs, on=['lonind', 'latind', 'year'], how = 'left')
cr = frpr[(frpr.fire > 0) & (frpr.peat > 0)].groupby('year')['repeat'].value_counts().reset_index(name = 'repeated')
cr['year'] = cr.year.astype(int)
yeard = {}
for year in range(2002, 2020, 1):
    #frps = comb[year].sum()
    reps = cr[cr.year == year][['repeat', 'repeated']]
    reps = reps.sort_values('repeat').reset_index()
    print(year)
    print(reps)
    #continue here
    reps.loc[5, 'repeated'] = reps.loc[ 5:, 'repeated'].sum()
    reps = reps.iloc[:6] / reps.sum()
    reps = reps.dropna()
    yeard[year] = reps['repeated'].values#/frps
yeard


"""
loss = pd.read_parquet('/mnt/data/forest/forest_loss_type_0.01deg_v3.parquet')
gri = Gridder(bbox = 'indonesia', step = 0.01)
loss = gri.add_coords_from_ind(loss)
loss = spatial_subset_dfr(loss, [5.8, 95, -5.935, 119])
loss = loss[~((loss.longitude > 118) & (loss.latitude < -2.5))]
loss_prim = loss.filter(like = '_loss_prim', axis = 1)
loss_sec = loss.filter(regex = '^(?=.*loss)(?!.*prim).*', axis = 1)
lp = loss_prim.sum(axis = 0)
lp.index = range(2001, 2019, 1)
lp = pd.DataFrame(lp, columns = ['loss_prim'])
lp['loss_sec'] = loss_sec.sum(axis = 0).values
lp['loss_total'] = lp['loss_prim'] + lp['loss_sec']
lp.to_parquet('../pyviz/data/loss_yearly.parquet')

#from notebook
elmask = (comb['1997'] > 0) | (comb['1998'] > 0)
el2mask = (comb['2015'] > 0)
el = comb[elmask].shape[0]
#print(comb[comb > 0].count())
print('Number of cells burned in 1997 - 1998:', el)

comb[comb > 0] = 1
print(comb.sum())
print(comb[(elmask) & (comb.iloc[:, 2:].sum(axis = 1) > 0)].sum())
print(comb[(el2mask) & (comb.iloc[:, :-4].sum(axis = 1) > 0)].sum())
"""
