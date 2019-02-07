import os
import datetime
import numpy as np
import xarray as xr
from netCDF4 import Dataset
#import geopandas as gpd
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
import cartopy.io.shapereader as shapereader
from mpl_toolkits.axes_grid1 import AxesGrid
from cartopy.feature import ShapelyFeature
from cartopy import feature
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.pyplot as plt
from cartopy.feature import NaturalEarthFeature


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

    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    dataset.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(),
                           x = 'longitude', y='latitude')
    gl = ax.gridlines(ccrs.PlateCarree(), draw_labels=True)
    gl.xlabels_top = False
    #borders = admin_borders()
    #ax.add_feature(borders)
    #ax.add_feature(coast)
    ax.coastlines('50m')
    plt.tight_layout(pad=2, w_pad=3, h_pad=7.0)
    #plt.title('FFMC vs FRP count break point')
    #plt.savefig('figs/25km_FFMC_break.png', dpi = 300, bbox_inches='tight')
    plt.show()

def mask_ocean(dar, land_mask):
    dar = dar.where(land_mask['lsm'][0, :, :].values)
    return dar

def spatial_subset(dataset, bbox):
    """
    Selects data within spatial bbox. bbox coords must be given as
    positive values for the Northern hemisphere, and negative for
    Southern. West and East both positive - Note - the method is
    naive and will only work for bboxes fully fitting in the Eastern hemisphere!!!
    Args:
        dataset - xarray dataset
        bbox - (list) [North, West, South, East]
    Returns:
        xarray dataset
    """
    dataset = dataset.sel(longitude = slice(bbox[1], bbox[3]))
    dataset = dataset.sel(latitude = slice(bbox[0], bbox[2]))
    return dataset


def make_seas5_fwi_plot(ds, bbox, prod):
    land_mask = 'data/era_land_mask.nc'
    land_mask = xr.open_dataset(land_mask)
    land_mask = spatial_subset(land_mask, bbox)
    land_mask = land_mask.interp_like(ds.isel(time=0))
    ds = ds[prod]
    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes,
                  dict(map_projection=projection))
    fig = plt.figure(figsize=(12,10))
    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(3, 2),
                    axes_pad=.3,
                    cbar_mode='edge',
                    cbar_location='bottom',
                    cbar_pad=0.5,
                    cbar_size='3%',
                    label_mode='')
    for nr, ax1 in enumerate(axgr, 0):
        ds_sel = ds.isel(time=nr)
        ds_sel = mask_ocean(ds_sel, land_mask)
        #if (month==1) and pn==0:
        #    ax1.set_title('ERA5 t2m anomaly')
        #elif (month==1) and pn==1:
        #    ax1.set_title('SEAS5 t2m anomaly 2m lead')
        vmin = ds.min()
        vmax = 200
        img = ds_sel.plot.pcolormesh(ax=ax1, transform=ccrs.PlateCarree(),
                               x = 'longitude', y='latitude',
                               vmin=vmin, vmax=vmax,
                               add_colorbar=False)#, add_labels=False)
        ax1.add_feature(cartopy.feature.COASTLINE)
        #    if pn == 0:
        #        ax1.set_ylabel('{0}/{1}'.format(year, month))
        #        ax1.set_yticks([-30], crs=projection)
        #        ax1.set_yticklabels([""])
        axgr.cbar_axes[0].colorbar(img)
        axgr.cbar_axes[1].colorbar(img)
        #axgr.cbar_axes[2].colorbar(img)
        #axgr.cbar_axes[3].colorbar(img2)
        #axgr.cbar_axes[4].colorbar(img2)
        #axgr.cbar_axes[5].colorbar(img2)
    tit = fig.suptitle('2018-11-01 mean monthly {0} forecast'.format(prod), y=.9, fontsize=18)
    plt.savefig('figs/2018-11_s5_{0}_indonesia.png'.format(prod), dpi=80, bbox_inches='tight', bbox_extra_artists=[tit])


