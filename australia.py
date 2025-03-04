import glob
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy
from envdata import *
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.feature import ShapelyFeature
import cartopy.io.shapereader as shapereader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from mpl_toolkits.axes_grid1 import AxesGrid

def austr_states():
    #peat_path = '/home/tadas/tofewsi/data/peat_atlas'
    #peat_fname = 'WI_PeatAtlas_SumatraKalimantan_MERGED_DTRV120914_without_legend_hapus2.shp'
    peat_path = 'data/borders'
    peat_fname = 'ne_10m_admin_1_states_provinces.shp'
    peat_shp = os.path.join(peat_path, peat_fname)
    peat_shapes = shapereader.Reader(peat_shp)
    geoms = peat_shapes.geometries()
    countries = peat_shapes.records()
    aust = [x[1] for x in zip(countries, geoms) if x[0].attributes['admin'] == 'Australia']
    austr_state_borders = ShapelyFeature(aust,
                               ccrs.PlateCarree(),
                               facecolor='none',
                               edgecolor='black',
                               alpha = 0.5)
    return austr_state_borders
 
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
    dataset = dataset.where((dataset[lat_name[0]] <= bbox[0]) &
                            (dataset[lat_name[0]] >= bbox[1]), drop=True)
    dataset = dataset.where((dataset[lon_name[0]] >= bbox[2]) &
                            (dataset[lon_name[0]] <= bbox[3]), drop=True)
    return dataset

def plot_dataset(dataset):
    fig = plt.figure(figsize=(10, 6))
    for nr, month in enumerate(mon['month'].values):
        ax = fig.add_subplot(4, 14, month, projection=ccrs.PlateCarree())
        mon[nr].plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(),
                                   x = 'longitude', y='latitude', add_colorbar=False)
        ax.add_feature(borders)
        #ax.add_feature(feature.BORDERS, linestyle='-')
        #ax.add_feature(feature.COASTLINE)
 
    fig = plt.figure(figsize=(10,6))
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
                           x = 'longitude', y='latitude')
    ax.add_feature(feature.BORDERS)
    ax.add_feature(feature.COASTLINE)
    ax.gridlines(ccrs.PlateCarree(), draw_labels=True)
    plt.tight_layout(pad=2, w_pad=3, h_pad=7.0)
    #plt.savefig('lulc_2010_riau.png', dpi = 80)
    plt.show()

#data_path = '/mnt/data/SEAS5/australia'
#d_range = pd.date_range('2009-06-01', periods=14, freq=pd.offsets.MonthBegin())
#fnames = glob.glob(os.path.join(data_path, '{0}*.*'.format(d_range[0].date())))

#store_name = os.path.join(data_path, '2013-05-01_164.128_165.128_166.128_167.128_168.128_169.128_228.128_0.25deg.nc')
#fo = Envdata(data_path, os.path.join(data_path, store_name))

#first climatology.
def get_era5_climatology(data_path):
    dps = []
    for year in range(2008, 2018, 1):
        """
        fname = glob.glob(os.path.join(data_path, '{0}*165.128*'.format(year)))[0]
        ds = xr.open_dataset(fname)
        ds = ds['t2m'][::6, :, :]
        dm = ds.groupby('time.month').max('time')
        dm.to_netcdf('data/aust_era5_{0}_t2m_mmax.nc'.format(ds.time.dt.year[0].values))
        #dps.append(ds)
        """
        fname = glob.glob(os.path.join(data_path, '{0}*228.128*'.format(year)))[0]
        dsp = xr.open_dataset(fname)
        dsp = dsp['tp']
        #dm = dsp.groupby('time.month').sum('time')
        #dm.to_netcdf('data/aust_era5_{0}_tp_mm.nc'.format(dsp.time.dt.year[0].values))
        dps.append(dsp)
    return dps

def read_monthly_means():
    t2m = xr.open_dataset('data/aust_mean_monthly_t2m.nc')
    t2max = xr.open_dataset('data/aust_max_monthly_t2m.nc')
    tpm = xr.open_dataset('data/aust_sum_mean_monthly_tp.nc')
    return t2m, t2max, tpm

def create_monthly_stats(year, month_leads):
    ds_list = []
    era5_t2m, era5_tp = era5_make_means(year)
    era5_t2m = era5_t2m.to_dataset(name = 'e5_t2m')
    era5_tp = era5_tp.to_dataset(name = 'e5_tp')
    ds_list.extend([era5_t2m, era5_tp])
    month_dates = pd.date_range('{0}-01-01'.format(year),
                                periods = 12, freq = pd.offsets.MonthBegin())
    for month_date in month_dates:
        for lead in month_leads:
            s5_t2m, s5_tp = seas5_make_means(month_date, lead)
            s5_t2m_name = 's5_t2m_{}'.format(lead)
            s5_t2m = xr.Dataset({s5_t2m_name: (('month', 'latitude', 'longitude'),
                                 np.expand_dims(s5_t2m.values, 0))},
                                 coords = {'month': [month_date.month],
                                           'latitude': s5_t2m.latitude,
                                           'longitude': s5_t2m.longitude})
            s5_tp_name = 's5_tp_{}'.format(lead)
            s5_tp = xr.Dataset({s5_tp_name: (('month', 'latitude', 'longitude'),
                                 np.expand_dims(s5_tp.values, 0))},
                                 coords = {'month': [month_date.month],
                                           'latitude': s5_tp.latitude,
                                           'longitude': s5_tp.longitude})
            ds_list.extend([s5_t2m, s5_tp])
    ds_all = xr.merge(ds_list)
    return ds_all

def seas5_make_means(month_date, month_lead):
    year = month_date.year
    month = month_date.month
    data_path = '/mnt/data/SEAS5/australia_2l'
    file_date = month_date - pd.DateOffset(months = month_lead)
    fname = glob.glob(os.path.join(data_path, '{0}*.*'.format(file_date.date())))[0]
    ds = xr.open_dataset(fname)
    ds = ds[['t2m', 'tp']]
    ds = ds.sel(time = ds['time.month'] == month_date.month)
    ensamble_median = ds.median('number')
    t2m_mean = ensamble_median['t2m'].mean('time')
    tp_sum = ensamble_median['tp'][-1, :, :] - ensamble_median['tp'][0, :, :]
    return t2m_mean, tp_sum

def era5_make_means(year):
    fname = glob.glob(os.path.join(data_path, '{0}*165.128*'.format(year)))[0]
    ds = xr.open_dataset(fname)
    ds = ds['t2m'][::6, :, :]
    t2m_mean = ds.groupby('time.month').mean('time')
    fname = glob.glob(os.path.join(data_path, '{0}*228.128*'.format(year)))[0]
    dsp = xr.open_dataset(fname)
    tp = dsp['tp'].groupby('time.month').sum('time')
    return t2m_mean, tp


def seas5_means(dt):
    year = dt.year
    month = dt.month
    dt = dt - pd.DateOffset(months = 3)
    t2m = xr.open_dataset('data/aust_s5_t2m_mm_{0}_{1}.nc'.format(year, month))
    tp = xr.open_dataset('data/aust_s5_tp_mm_{0}_{1}.nc'.format(year, month))
    return t2m, tp

def era5_mean(dt):
    year = dt.year
    month = dt.month
    data_path = '/mnt/data/era5/australia'
    t2m = xr.open_dataset('data/aust_era5_{0}_t2m_mm.nc'.format(year))
    t2m = t2m.sel(month=month)
    t2max = xr.open_dataset('data/aust_era5_{0}_t2m_mmax.nc'.format(year))
    t2max = t2max.sel(month=month)
    tpm = xr.open_dataset('data/aust_era5_{0}_tp_mm.nc'.format(year))
    tpm = tpm.sel(month=month)
    return t2m, t2max, tpm

def mask_ocean(dar, land_mask):
    dar = dar.where(land_mask['lsm'][0, :, :].values)
    return dar

def do_plots(dates, t2m, t2max, tpm, land_mask):
    plot_nr = len(dates)
    fig = plt.figure(figsize = (25, 3.5 * plot_nr))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for nr, dt in enumerate(dates):
        year = dt.year
        month = dt.month
        et2m, et2max, etpm = era5_mean(dt)
        st2m, stpm = seas5_means(dt)
        at2m = et2m - t2m.sel(month=month)
        ast2m = st2m - t2m.sel(month=month)
        atpm = tpm.sel(month=month) - etpm
        astpm = tpm.sel(month=month) - stpm
        for pn, item in enumerate([mask_ocean(at2m['t2m'], land_mask),
                                   mask_ocean(ast2m['t2m'], land_mask),
                                   mask_ocean(atpm['tp'], land_mask),
                                   mask_ocean(astpm['tp'], land_mask)]):
            ax1 = plt.subplot2grid((plot_nr, 4), (nr, pn), projection=ccrs.PlateCarree())
            #item.plot()
            if pn < 2:
                vmin = -4.5
                vmax = 4.5
            if pn > 1:
                vmin = -.1
                vmax = .1
            item.plot.pcolormesh(ax=ax1, transform=ccrs.PlateCarree(),
                                   x = 'longitude', y='latitude', cmap='seismic', vmin=vmin, vmax=vmax, add_colorbar=True)
            #ax1.add_feature(cartopy.feature.OCEAN)
            ax1.add_feature(cartopy.feature.COASTLINE)
            if (pn==0):
                ax1.set_title('ERA5 mean temp anomaly {0}/{1}'.format(year, month))
            elif (pn==1):
                ax1.set_title('SEA5 mean temp anomaly 3 month lead {0}/{1}'.format(year, month))
            elif (pn==2):
                ax1.set_title('ERA5 precipitation anomaly {0}/{1}'.format(year, month))
            elif (pn==3):
                ax1.set_title('SEAS5 precipitation anomaly 3 month lead {0}/{1}'.format(year, month))
            if pn == 0:
                ax1.set_ylabel('{0} {1}'.format(year, month))
    #plt.tight_layout()
    fig.suptitle('2009-09 to 2010-10', fontsize=32)
    plt.savefig('aust_t2m_mm_2009.png', dpi=80)

def do_plots_year(year, lead, t2m, t2max, tpm, land_mask, borders):
    ds = xr.open_dataset('data/aust_mm_{0}.nc'.format(year))
    projection = ccrs.PlateCarree() 
    axes_class = (GeoAxes, 
                  dict(map_projection=projection)) 
    fig = plt.figure(figsize=(33,40)) 
    axgr = AxesGrid(fig, 111, axes_class=axes_class, 
                    nrows_ncols=(12, 6), 
                    axes_pad=.3, 
                    cbar_mode='edge', 
                    cbar_location='bottom',
                    cbar_pad=0.5, 
                    cbar_size='3%', 
                    label_mode='')    
    for month, row in enumerate(axgr.axes_row, 1):
        et2m = ds['e5_t2m'].sel(month = month)
        etpm = ds['e5_tp'].sel(month = month) * 1000
        st2m2 = ds['s5_t2m_2'].sel(month = month)
        st2m3 = ds['s5_t2m_3'].sel(month = month)
        stpm2 = ds['s5_tp_2'].sel(month = month) * 1000
        stpm3 = ds['s5_tp_3'].sel(month = month) * 1000

        at2m = et2m - t2m.sel(month=month)
        ast2m2 = st2m2 - t2m.sel(month=month)
        ast2m3 = st2m3 - t2m.sel(month=month)
        atpm = etpm - tpm.sel(month=month) * 1000
        astpm2 = stpm2 - tpm.sel(month=month) * 1000
        astpm3 = stpm3 - tpm.sel(month=month) * 1000
        for pn, item in enumerate([mask_ocean(at2m['t2m'], land_mask),
                                   mask_ocean(ast2m2['t2m'], land_mask),
                                   mask_ocean(ast2m3['t2m'], land_mask),
                                   mask_ocean(atpm['tp'], land_mask),
                                   mask_ocean(astpm2['tp'], land_mask),
                                   mask_ocean(astpm3['tp'], land_mask)]):
            ax1 = row[pn]
            if (month==1) and pn==0:
                ax1.set_title('ERA5 t2m anomaly')
            elif (month==1) and pn==1:
                ax1.set_title('SEAS5 t2m anomaly 2m lead')
            elif (month==1) and pn==2:
                ax1.set_title('SEAS5 t2m anomaly 3m lead')
            elif (month==1) and pn==3:
                ax1.set_title('ERA5 tp anomaly')
            elif (month==1) and pn==4:
                ax1.set_title('SEAS5 tp anomaly 2m lead')
            elif (month==1) and pn==5:
                ax1.set_title('SEAS5 tp anomaly 3m lead')
            else:
                ax1.set_title('')
            if pn < 3:
                vmin = -5
                vmax = 5
                img = item.plot.pcolormesh(ax=ax1, transform=ccrs.PlateCarree(),
                                       x = 'longitude', y='latitude',
                                       cmap='seismic', vmin=vmin, vmax=vmax,
                                       add_colorbar=False, add_labels=False)
                ax1.add_feature(cartopy.feature.COASTLINE)
            else:
                vmin = - 360
                vmax = 360
                img2 = item.plot.pcolormesh(ax=ax1, transform=ccrs.PlateCarree(),
                                       x = 'longitude', y='latitude',
                                       cmap='seismic_r', vmin=vmin, vmax=vmax,
                                       add_colorbar=False, add_labels=False)
                ax1.add_feature(cartopy.feature.COASTLINE)

            if pn == 0:
                ax1.set_ylabel('{0}/{1}'.format(year, month))
                ax1.set_yticks([-30], crs=projection)
                ax1.set_yticklabels([""])
                """
                if (pn==0):
                    ax1.set_title('ERA5 t2m anomaly)
                elif (pn==1):
                    ax1.set_title('SEA5 mean temp anomaly 3 month lead {0}/{1}'.format(year, month))
                elif (pn==2):
                    ax1.set_title('ERA5 precipitation anomaly {0}/{1}'.format(year, month))
                elif (pn==3):
                    ax1.set_title('SEAS5 precipitation anomaly 3 month lead {0}/{1}'.format(year, month))
                if pn == 0:
                    ax1.set_ylabel('{0} {1}'.format(year, month))
                """
            ax1.add_feature(borders)
        axgr.cbar_axes[0].colorbar(img)
        axgr.cbar_axes[0].set_title('Degrees C')
        axgr.cbar_axes[1].colorbar(img)
        axgr.cbar_axes[1].set_title('Degrees C')
        axgr.cbar_axes[2].colorbar(img)
        axgr.cbar_axes[2].set_title('Degrees C')
        axgr.cbar_axes[3].colorbar(img2)
        axgr.cbar_axes[3].set_title('mm')
        axgr.cbar_axes[4].colorbar(img2)
        axgr.cbar_axes[4].set_title('mm')
        axgr.cbar_axes[5].colorbar(img2)
        axgr.cbar_axes[5].set_title('mm')
    tit = fig.suptitle('Year {0}'.format(year), y=.9, fontsize=32)
    plt.savefig('Australia_{0}_state_borders_r.png'.format(year), dpi=80, bbox_inches='tight', bbox_extra_artists=[tit])




#TODO
#for year make plots: era5 t anomaly
land_mask = 'data/era_land_mask.nc'
land_mask = xr.open_dataset(land_mask)
australia = [-10, -44, 113, 154]
land_mask = spatial_subset(land_mask, australia)
#data_path = '/mnt/data/SEAS5/australia'
data_path = '/mnt/data/era5/australia'
#ds = create_monthly_stats(2009, [2,3])
t2m, t2max, tpm = read_monthly_means()
#dates = pd.date_range('2009-09-01', periods=14, freq=pd.offsets.MonthBegin())
#for year in range(2009, 2012, 1):
borders = austr_states()
do_plots_year(2011, 2, t2m, t2max, tpm/10, land_mask, borders)
#do_plots(dates, t2m, t2max, tpm, land_mask)
#for dt in dates:
#    seas5_make_means(dt)

