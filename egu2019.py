import io
import pandas as pd
import xarray as xr
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
                               edgecolor='black',
                               alpha = 0.5)
    return ind_state_borders

def plot_ind_data(indon, dataset):
    ax = plt.axes(projection=ccrs.PlateCarree())
    #ax.add_feature(feature.COASTLINE)
    ax.add_feature(indon)
    ax.set_extent((94.8, 141.1, 6.2, -10.5))
    ax.outline_patch.set(visible=False)
    plt.savefig("test.svg", format="svg")
    plt.show()

def get_probs(dfr, model, year, month):
    probs = dfr[model][(dfr.year == year) & (dfr.month == month)]
    return probs


gri = Gridder(bbox = 'indonesia', step = 0.25)
probs = pd.read_parquet('data/feature_frame_0.25deg_v2.parquet')
indon = ind_states()
plot_ind_data(indon)
