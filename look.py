import os
import datetime
import numpy as np
import xarray as xr
from netCDF4 import Dataset
#import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.io.shapereader as shapereader
from cartopy.feature import ShapelyFeature
from cartopy import feature
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.pyplot as plt

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    #return base.from_list(cmap_name, color_list, N)
    return LinearSegmentedColormap.from_list(cmap_name, color_list, N)

def peatlands():
    #peat_path = '/home/tadas/tofewsi/data/peat_atlas'
    #peat_fname = 'WI_PeatAtlas_SumatraKalimantan_MERGED_DTRV120914_without_legend_hapus2.shp'
    peat_path = '/mnt/data/land_cover/peatlands/'
    peat_fname = 'Peatland_land_cover_2015'
    peat_shp = os.path.join(peat_path, peat_fname)
    peat_shapes = shapereader.Reader(peat_shp).geometries()
    peatlands = ShapelyFeature(peat_shapes,
                               ccrs.PlateCarree(),
                               facecolor='none',
                               edgecolor='black',
                               alpha = 0.5)
    return peatlands
 

def admin_borders():
    ind_bord_path = '/home/tadas/tofewsi/data/ind_adm_borders'
    ind_shp = os.path.join(ind_bord_path, 'IDN_adm1.shp')
    ind_bords = shapereader.Reader(ind_shp).geometries()
    borders = ShapelyFeature(ind_bords,
                             ccrs.PlateCarree(),
                             facecolor='none',
                             edgecolor='black',
                             alpha = 0.5)
    return borders

def geopandas_read_shp():
    peat_path = '/mnt/data/land_cover/peatlands/'
    peat_fname = 'Peatland_land_cover_2015.shp'
    #peat_path = '/home/tadas/tofewsi/data/peat_atlas'
    #peat_fname = 'WI_PeatAtlas_SumatraKalimantan_MERGED_DTRV120914_without_legend_hapus2.shp'
    peat_fr = gpd.read_file(os.path.join(peat_path, peat_fname))
    not_valid = []
    for nr, row in peat_fr.iterrows():
        if not row['geometry'].is_valid:
            not_valid.append(nr)
    peat_fr = peat_fr[~peat_fr.index.isin(not_valid)]
    return peat_fr


def read_peatland_lc():
    data_path = '/mnt/data/land_cover/peatlands'
    products = []
    years = [1990, 2007, 2015]
    sp_res = 0.05
    for year in years:
        fname = 'peatland_lc_{0}.tif'.format(year)
        fname_path = os.path.join(data_path, fname)
        product = xr.open_rasterio(fname_path)
        products.append(product.values.squeeze())
        lons = product.x.values
        #latitudes need tiying as they are off by a bit
        #porbably due to float conversion somewhere in the pipline
        lats = np.arange(product.y[-1].values.round(0) + sp_res/2, 
                         product.y[0].values + sp_res/2, 
                         sp_res)[::-1]
    dataset = xr.Dataset({ 'lc': (('year', 'latitude', 'longitude'),
                                        np.array(products))},
                  coords = {'year': [str(x) for x in years],
                             'latitude': lats,
                            'longitude': lons})
    return dataset


def plot_indonesia_discrete():
    fig = plt.figure(figsize=(8, 17))
    #ax = fig.add_subplot(projection=ccrs.PlateCarree()) #ax = plt.axes(projection=ccrs.PlateCarree()) #borders = admin_borders()
    peats = read_peatland_lc()
    class_nr = np.unique(peats.lc).shape[0]
    cmap_ = ListedColormap(['royalblue', 'cyan', 'yellow', 'orange', 'darkorchid', 'goldenrod', 
                                      'forestgreen', 'turquoise', 'indigo', 'hotpink', 'crimson', 'peru',
                                      'navajowhite', 'khaki', 'lime', 'steelblue'])
    #cmap_ = matplotlib.cm.tab20

    #cmap_ = discrete_cmap(class_nr, base_cmap = 'Spectral')
    #cmap_ ''= discrete_cmap(class_nr, base_cmap = 'terrain')
    cmap_.set_under('white')
    for nr, year in enumerate(['1990', '2007', '2015'], 1):
        ax = fig.add_subplot(3, 1, nr, projection=ccrs.PlateCarree())
        im = peats.sel(year=year)['lc'].plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(),
                               x = 'longitude', y='latitude',cmap=cmap_,vmin=1, vmax=16, add_colorbar=True)
        #ax.add_feature(borders)
        #ax.add_feature(im)
        gl = ax.gridlines(ccrs.PlateCarree(), draw_labels=True)
        gl.xlabels_top = gl.ylabels_right = False
        #ax.set_title(lc_name)
            #plt.colorbar(im, ax=ax, shrink=.62, orientation='horizontal')
     
        #ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        #dataset.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(),
        #                       x = 'longitude', y='latitude', add_colorbar=False)
        #ax.gridlines(ccrs.PlateCarree(), draw_labels=True)
        #ax.add_feature(borders)
        ax.add_feature(feature.COASTLINE)
    plt.tight_layout(pad=2, w_pad=3, h_pad=7.0)
    #plt.savefig('peatlands_lc.png', dpi = 80)
    plt.show()

def plot_peatlands(dataset):
    fig = plt.figure(figsize=(18,6))
    for nr, month in enumerate(mon['month'].values):
        ax = fig.add_subplot(3, 4, month, projection=ccrs.PlateCarree())
        mon[nr].plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(),
                                   x = 'longitude', y='latitude', add_colorbar=False)
        ax.add_feature(borders)
        #ax.add_feature(feature.BORDERS, linestyle='-')
        #ax.add_feature(feature.COASTLINE)
 
def plot_dataset(dataset):
    fig = plt.figure(figsize=(18,6))
    #select hour
    #ds.sel(time=datetime.time(1))
    """
    for nr, month in enumerate(mon['month'].values):
        ax = fig.add_subplot(3, 4, month, projection=ccrs.PlateCarree())
        mon[nr].plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(),
                                   x = 'longitude', y='latitude', add_colorbar=False)
        ax.add_feature(borders)
        #ax.add_feature(feature.BORDERS, linestyle='-')
        #ax.add_feature(feature.COASTLINE)
    lc_names = ['Majority_Land_Cover_Type_1',
                'Majority_Land_Cover_Type_2',
                'Majority_Land_Cover_Type_3']

    for nr, lc_name in enumerate(lc_names, 1):
        ax = fig.add_subplot(1, 3, nr, projection=ccrs.PlateCarree())
        ll = dataset[lc_name]
        im = ll.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(),
                           x = 'longitude', y='latitude', cmap=discrete_cmap(14), add_colorbar=False)
        gl = ax.gridlines(ccrs.PlateCarree(), draw_labels=True)
        gl.xlabels_top = gl.ylabels_right = False
        ax.add_feature(borders)
        ax.set_title(lc_name)
        #plt.colorbar(im, ax=ax, shrink=.62, orientation='horizontal')
 
    """
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    dataset.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(),
                           x = 'longitude', y='latitude', add_colorbar=False)
    ax.gridlines(ccrs.PlateCarree(), draw_labels=True)
    #borders = admin_borders()
    #ax.add_feature(borders)
    ax.add_feature(feature.COASTLINE)
    plt.tight_layout(pad=2, w_pad=3, h_pad=7.0)
    #plt.savefig('lulc_2010_riau.png', dpi = 80)
    plt.show()

