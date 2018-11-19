import glob
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
from envdata import *

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

def seas5_make_means(dt):
    year = dt.year
    month = dt.month
    dt = dt - pd.DateOffset(months = 3)
    print(dt)
    data_path = '/mnt/data/SEAS5/australia'
    fnames = glob.glob(os.path.join(data_path, '{0}*.*'.format(dt.date())))
    ds = xr.open_dataset(fnames[0])
    stp = ds['t2m'].median('number')
    st2m = stp.mean('time')
    st2m.to_netcdf('data/aust_s5_t2m_mm_{0}_{1}.nc'.format(year, month))
    #stp = ds['tp'].median('number')
    #stp = ds['tp'].mean('number')
    #stp = stp[-1,:,:] - stp[0,:,:]
    #stp_day = stp.groupby('time.day').sum('time')
    #st2m = ds['t2m'].mean(['number', 'time'])
    #st2m.to_netcdf('data/aust_s5_t2m_mm_{0}_{1}.nc'.format(year, month))
    #stp.to_netcdf('data/aust_s5_tp_mm_{0}_{1}.nc'.format(year, month))
    #return st2m, stp

def era5_make_means(dt):
    d
    dps = []
    for year in range(2008, 2018, 1):
        """
        fname = glob.glob(os.path.join(data_path, '{0}*165.128*'.format(year)))[0]
        ds = xr.open_dataset(fname)
        ds = ds['t2m'][::6, :, :]
        dm = ds.groupby('time.month').mean('time')
        dm.to_netcdf('data/aust_era5_{0}_t2m_mm.nc'.format(ds.time.dt.year[0].values))
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



land_mask = 'data/era_land_mask.nc'
land_mask = xr.open_dataset(land_mask)
australia = [-10, -44, 113, 154]
land_mask = spatial_subset(land_mask, australia)
#data_path = '/mnt/data/SEAS5/australia'
data_path = '/mnt/data/era5/australia'
t2m, t2max, tpm = read_monthly_means()
dates = pd.date_range('2009-09-01', periods=14, freq=pd.offsets.MonthBegin())
do_plots(dates, t2m, t2max, tpm, land_mask)
#for dt in dates:
#    seas5_make_means(dt)



