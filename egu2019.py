import io
import pandas as pd
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy import feature
import cartopy.io.shapereader as shapereader
from gridding import Gridder

def ind_states():
    #peat_path = '/home/tadas/tofewsi/data/peat_atlas'
    #peat_fname = 'WI_PeatAtlas_SumatraKalimantan_MERGED_DTRV120914_without_legend_hapus2.shp'
    #ind_shp = 'data/borders/ne_10m_admin_1_states_provinces.shp'
    ind_shp = 'data/borders/ne_50m_admin_0_countries.shp'
    ind_shapes = shapereader.Reader(ind_shp)
    geoms = ind_shapes.geometries()
    countries = list(ind_shapes.records())
    indo = [x for x in zip(countries, geoms) if 'Indonesia' in x[0].attributes['ADMIN']]
    #indo = [x for x in zip(countries, geoms) if 'Indonesia' in x[0].attributes['admin']]
    ind_state_borders = feature.ShapelyFeature(indo[0][1],
                               ccrs.PlateCarree(),
                               facecolor='none',
                               edgecolor='grey',
                               alpha = 1)
    return ind_state_borders

def plot_ind_data(indon, dataset, variable):
    fig = plt.figure(figsize=(35,10))
    ax = fig.add_axes([0, 0, 1, 1], projection = ccrs.PlateCarree())
    #ax.add_feature(feature.COASTLINE)
    ax.add_feature(indon, zorder=0)
    ax.set_extent((94.8, 141.1, 6.2, -10.5))
    ax.outline_patch.set(visible=False)
    pcm = ax.pcolormesh(ds.longitude - (gri.step / 2), ds.latitude - (gri.step / 2), ds[variable],
                 transform=ccrs.PlateCarree(), cmap="inferno")
    plt.colorbar(pcm, orientation = 'horizontal', fraction = 0.03, pad = 0)
    ax.background_patch.set_alpha(0)
    ax.patch.set_alpha(0)
    plt.savefig("test.png", format="png", dpi=300, 
            bbox_inches='tight', pad_inches = 0, transparent = True)
    plt.show()

def get_probs(dfr, model, year, month):
    probs = dfr[[model, 'lonind', 'latind']][(dfr.year == year) & (dfr.month == month)]
    return probs


gri = Gridder(bbox = 'indonesia', step = 0.25)
dfr = pd.read_parquet('data/prob_dfr.parquet')
variable = 'NeuralNet_prob'
probs = get_probs(dfr, variable, 2015, 10)
ds = gri.dfr_to_dataset(probs, variable, np.nan)
indon = ind_states()
plot_ind_data(indon, ds, variable)

