import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import numpy as np
import pandas as pd
import xarray as xr
from envdata import Envdata

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
    print(lat_name, bbox)
    dataset = dataset.where((dataset[lat_name[0]] < bbox[0]) &
                            (dataset[lat_name[0]] > bbox[1]), drop=True)
    dataset = dataset.where((dataset[lon_name[0]] > bbox[2]) &
                            (dataset[lon_name[0]] < bbox[3]), drop=True)
    return dataset


def spatial_subset_dfr(dfr, bbox):
    """
    Selects data within spatial bbox. bbox coords must be given as
    positive values for the Northern hemisphere, and negative for
    Southern. West and East both positive - Note - the method is
    naive and will only work for bboxes fully fitting in the Eastern hemisphere!!!
    Args:
        dfr - pandas dataframe
        bbox - (list) [North, South, West, East]
    Returns:
        pandas dataframe
    """
    dfr = dfr[(dfr['lat'] < bbox[0]) &
                            (dfr['lat'] > bbox[1])]
    dfr = dfr[(dfr['lon'] > bbox[2]) &
                            (dfr['lon'] < bbox[3])]
    return dfr


def ds_monthly_means(darray, land_mask):
    darray_m = darray.groupby('time.month').mean() 
    darray_masked = darray_m.where(land_mask.values)
    return darray_masked


def dfr_monthly_counts(dfr):
    dfr_m = dfr.day_since.groupby([dfr.date.dt.year, 
                                   dfr.date.dt.month]).count().mean(level=1)
    return dfr_m 



def plot_comp_nc(fwi, dataset, bboxes, land_mask, suptitle, y2_label, figname):
    fig = plt.figure(figsize=(19,10))

    fwi15 = fwi.sel(time = '2015')
    ds15 = dataset.sel(time = '2015')

    fwi_m = ds_monthly_means(spatial_subset(fwi, bboxes['Indonesia']), land_mask)
    ba_m = ds_monthly_means(spatial_subset(dataset, bboxes['Indonesia']), land_mask)
    ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=1)
    months = list(range(1, 13, 1))
    ax1.plot(months, fwi_m.values, 'b-')
    ax1.set_xlabel('Month')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Mean FWI 2008 - 2015' , color='b')
    ax1.tick_params('y', colors='b')
    ax12 = ax1.twinx()
    ax12.set_ylabel('Mean ' + y2_label + ' 2008 - 2015', color='r')
    print(ba_m)
    ax12.bar(months, ba_m, color='r', alpha=.6)
    ax12.tick_params('y', colors='r')
    ax1.set_title(list(bboxes.keys())[0])

    ax2 = plt.subplot2grid((2, 4), (0, 1), colspan=1)
    fwi_m = ds_monthly_means(spatial_subset(fwi, bboxes['Kalimantan']), land_mask)
    ba_m = ds_monthly_means(spatial_subset(dataset, bboxes['Kalimantan']), land_mask)
    print(bboxes['Kalimantan'])
    ax2.plot(months, fwi_m.values, 'b-')
    ax2.set_xlabel('Month')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax2.set_ylabel('Mean FWI 2008 - 2015', color='b')
    ax2.tick_params('y', colors='b')
    ax22 = ax2.twinx()
    ax22.bar(months, ba_m, color='r', alpha=.6)
    ax22.set_ylabel('Mean ' + y2_label + ' 2008 - 2015', color='r')
    ax22.tick_params('y', colors='r')
    ax2.set_title(list(bboxes.keys())[1])


    ax3 = plt.subplot2grid((2, 4), (0, 2), colspan=1)
    fwi_m = ds_monthly_means(spatial_subset(fwi, bboxes['South Sumatra']), land_mask)
    ba_m = ds_monthly_means(spatial_subset(dataset, bboxes['South Sumatra']), land_mask)
    print(bboxes['South Sumatra'])
    ax3.plot(months, fwi_m.values, 'b-')
    ax3.set_xlabel('Month')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax3.set_ylabel('Mean FWI 2008 - 2015', color='b')
    ax3.tick_params('y', colors='b')
    ax32 = ax3.twinx()
    ax32.bar(months, ba_m, color='r', alpha=.6)
    ax32.set_ylabel('Mean ' + y2_label+ ' 2008 - 2015', color='r')
    ax32.tick_params('y', colors='r')
    ax3.set_title(list(bboxes.keys())[2])

    ax4 = plt.subplot2grid((2, 4), (0, 3), colspan=1)
    fwi_m = ds_monthly_means(spatial_subset(fwi, bboxes['Inner Riau']), land_mask)
    ba_m = ds_monthly_means(spatial_subset(dataset, bboxes['Inner Riau']), land_mask)
    ax4.plot(months, fwi_m.values, 'b-')
    ax4.set_xlabel('Month')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax4.set_ylabel('Mean FWI 2008 - 2015', color='b')
    ax4.tick_params('y', colors='b')
    ax42 = ax4.twinx()
    ax42.bar(months, ba_m, color='r', alpha=.6)
    ax42.set_ylabel('Mean ' + y2_label+ ' 2008 - 2015', color='r')
    ax42.tick_params('y', colors='r')
    ax4.set_title(list(bboxes.keys())[3])

    #2015
    fwi_m = ds_monthly_means(spatial_subset(fwi15, bboxes['Indonesia']), land_mask)
    ba_m = ds_monthly_means(spatial_subset(ds15, bboxes['Indonesia']), land_mask)
    ax11 = plt.subplot2grid((2, 4), (1, 0), colspan=1)
    months = list(range(1, 13, 1))
    ax11.plot(months, fwi_m.values, 'b-')
    ax11.set_xlabel('Month')
    # Make the y-ax1is label, ticks and tick labels match the line color.
    ax11.set_ylabel('Mean FWI 2015', color='b')
    ax11.tick_params('y', colors='b')
    ax112 = ax11.twinx()
    ax112.set_ylabel(y2_label + ' 2015', color='r')
    ax112.bar(months, ba_m, color='r', alpha=.6)
    ax112.tick_params('y', colors='r')
    ax11.set_title(list(bboxes.keys())[0])

    ax12 = plt.subplot2grid((2, 4), (1, 1), colspan=1)
    fwi_m = ds_monthly_means(spatial_subset(fwi15, bboxes['Kalimantan']), land_mask)
    ba_m = ds_monthly_means(spatial_subset(ds15, bboxes['Kalimantan']), land_mask)
    print(bboxes['Kalimantan'])
    ax12.plot(months, fwi_m.values, 'b-')
    ax12.set_xlabel('Month')
    # Make the y-ax1is label, ticks and tick labels match the line color.
    ax12.set_ylabel('Mean FWI 2015', color='b')
    ax12.tick_params('y', colors='b')
    ax122 = ax12.twinx()
    ax122.bar(months, ba_m, color='r', alpha=.6)
    ax122.set_ylabel(y2_label + ' 2015', color='r')
    ax122.tick_params('y', colors='r')
    ax12.set_title(list(bboxes.keys())[1])


    ax13 = plt.subplot2grid((2, 4), (1, 2), colspan=1)
    fwi_m = ds_monthly_means(spatial_subset(fwi15, bboxes['South Sumatra']), land_mask)
    ba_m = ds_monthly_means(spatial_subset(ds15, bboxes['South Sumatra']), land_mask)
    print(bboxes['South Sumatra'])
    ax13.plot(months, fwi_m.values, 'b-')
    ax13.set_xlabel('Month')
    # Make the y-ax1is label, ticks and tick labels match the line color.
    ax13.set_ylabel('Mean FWI 2015', color='b')
    ax13.tick_params('y', colors='b')
    ax132 = ax13.twinx()
    ax132.bar(months, ba_m, color='r', alpha=.6)
    ax132.set_ylabel(y2_label + ' 2015', color='r')
    ax132.tick_params('y', colors='r')
    ax13.set_title(list(bboxes.keys())[2])

    ax14 = plt.subplot2grid((2, 4), (1, 3), colspan=1)
    fwi_m = ds_monthly_means(spatial_subset(fwi15, bboxes['Inner Riau']), land_mask)
    ba_m = ds_monthly_means(spatial_subset(ba15, bboxes['Inner Riau']), land_mask)
    ax14.plot(months, fwi_m.values, 'b-')
    ax14.set_xlabel('Month')
    # Make the y-ax1is label, ticks and tick labels match the line color.
    ax14.set_ylabel('Mean FWI 2015', color='b')
    ax14.tick_params('y', colors='b')
    ax142 = ax14.twinx()
    ax142.bar(months, ba_m, color='r', alpha=.6)
    ax142.set_ylabel(y2_label + ' 2015', color='r')
    ax142.tick_params('y', colors='r')
    ax14.set_title(list(bboxes.keys())[3])


    fig.suptitle(suptitle, size=16)
    #fig.suptitle('MODIS FRP Collection 6', size=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(fig_name, res=300)
    plt.show()


def plot_comp_gen(ds, ba, bboxes, land_mask, ds_label, suptitle, y2_label, fig_name):
    plot_nr = len(bboxes)
    months = np.array(list(range(1, 13, 1)))
    bar_width = 0.45
    fig = plt.figure(figsize = (4.8 * plot_nr, 5))
    fwi = ds[ds_label]
    fwi15 = fwi.sel(time = '2015')
    ba15 = ba[ba.date.dt.year == 2015]
    bbox_names = list(bboxes.keys())
    for nr, (key, bbox) in enumerate(bboxes.items()):
        ax1 = plt.subplot2grid((1, plot_nr), (0, nr), colspan=1)
        fwi_m = ds_monthly_means(spatial_subset(fwi, bbox), land_mask)
        fwi15_m = ds_monthly_means(spatial_subset(fwi15, bbox), land_mask)
        ba_m = dfr_monthly_counts(spatial_subset_dfr(ba, bbox))
        line = ax1.plot(months, fwi_m.values, 'b--')
        line15 = ax1.plot(months, fwi15_m.values, 'r-')
        ax1.set_xlabel('Month')
        ax1.set_ylabel(ds_label.upper())
        ba_m = dfr_monthly_counts(spatial_subset_dfr(ba, bbox))
        ba15_m = dfr_monthly_counts(spatial_subset_dfr(ba15, bbox))
        ax12 = ax1.twinx()
        ax12.set_ylabel(y2_label)
        bars_m = ax12.bar(ba_m.index.values, ba_m, bar_width, color='b', alpha=.4)
        bars15_m = ax12.bar(ba15_m.index.values + bar_width, ba15_m, bar_width, color='r', alpha=.6)
        ax1.set_title(key)
        ax1.set_xticks(months + bar_width / 2)
        ax1.set_xticklabels(months)
        custom_objs = [Line2D([0], [0], color='b'), Line2D([0], [0], color='r'), bars_m, bars15_m]
        ax1.legend((custom_objs),
                   ('Mean {0} 2008 - 2015'.format(ds_label), '{0} 2015'.format(ds_label),
                    'Mean {0} 2008 - 2015'.format(y2_label), '{0} 2015'.format(y2_label)))
        #fig.suptitle('MODIS FRP Collection 6', size=16)
    fig.suptitle(suptitle, size=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join('./figures', fig_name), res=300)
    plt.show()



def plot_comp(fwi, ba, bboxes, land_mask, suptitle, y2_label, fig_name):
    fig = plt.figure(figsize=(19,10))

    fwi15 = fwi.sel(time = '2015')
    ba15 = ba[ba.date.dt.year == 2015]

    fwi_m = ds_monthly_means(spatial_subset(fwi, bboxes['Indonesia']), land_mask)
    ba_m = dfr_monthly_counts(spatial_subset_dfr(ba, bboxes['Indonesia']))
    ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=1)
    print(fwi_m)
    months = list(range(1, 13, 1))
    ax1.plot(months, fwi_m.values, 'b-')
    ax1.set_xlabel('Month')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Mean FWI 2008 - 2015' , color='b')
    ax1.tick_params('y', colors='b')
    ax12 = ax1.twinx()
    ax12.set_ylabel('Mean ' + y2_label + ' 2008 - 2015', color='r')
    ax12.bar(ba_m.index.values, ba_m, color='r', alpha=.6)
    ax12.tick_params('y', colors='r')
    ax1.set_title(list(bboxes.keys())[0])

    ax2 = plt.subplot2grid((2, 4), (0, 1), colspan=1)
    fwi_m = ds_monthly_means(spatial_subset(fwi, bboxes['Kalimantan']), land_mask)
    ba_m = dfr_monthly_counts(spatial_subset_dfr(ba, bboxes['Kalimantan']))
    print(bboxes['Kalimantan'])
    ax2.plot(months, fwi_m.values, 'b-')
    ax2.set_xlabel('Month')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax2.set_ylabel('Mean FWI 2008 - 2015', color='b')
    ax2.tick_params('y', colors='b')
    ax22 = ax2.twinx()
    ax22.bar(ba_m.index.values, ba_m, color='r', alpha=.6)
    ax22.set_ylabel('Mean ' + y2_label + ' 2008 - 2015', color='r')
    ax22.tick_params('y', colors='r')
    ax2.set_title(list(bboxes.keys())[1])


    ax3 = plt.subplot2grid((2, 4), (0, 2), colspan=1)
    fwi_m = ds_monthly_means(spatial_subset(fwi, bboxes['South Sumatra']), land_mask)
    ba_m = dfr_monthly_counts(spatial_subset_dfr(ba, bboxes['South Sumatra']))
    print(bboxes['South Sumatra'])
    ax3.plot(months, fwi_m.values, 'b-')
    ax3.set_xlabel('Month')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax3.set_ylabel('Mean FWI 2008 - 2015', color='b')
    ax3.tick_params('y', colors='b')
    ax32 = ax3.twinx()
    ax32.bar(ba_m.index.values, ba_m, color='r', alpha=.6)
    ax32.set_ylabel('Mean ' + y2_label+ ' 2008 - 2015', color='r')
    ax32.tick_params('y', colors='r')
    ax3.set_title(list(bboxes.keys())[2])

    ax4 = plt.subplot2grid((2, 4), (0, 3), colspan=1)
    fwi_m = ds_monthly_means(spatial_subset(fwi, bboxes['Inner Riau']), land_mask)
    ba_m = dfr_monthly_counts(spatial_subset_dfr(ba, bboxes['Inner Riau']))
    ax4.plot(months, fwi_m.values, 'b-')
    ax4.set_xlabel('Month')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax4.set_ylabel('Mean FWI 2008 - 2015', color='b')
    ax4.tick_params('y', colors='b')
    ax42 = ax4.twinx()
    ax42.bar(ba_m.index.values, ba_m, color='r', alpha=.6)
    ax42.set_ylabel('Mean ' + y2_label+ ' 2008 - 2015', color='r')
    ax42.tick_params('y', colors='r')
    ax4.set_title(list(bboxes.keys())[3])

    #2015
    fwi_m = ds_monthly_means(spatial_subset(fwi15, bboxes['Indonesia']), land_mask)
    ba_m = dfr_monthly_counts(spatial_subset_dfr(ba15, bboxes['Indonesia']))
    ax11 = plt.subplot2grid((2, 4), (1, 0), colspan=1)
    months = list(range(1, 13, 1))
    ax11.plot(months, fwi_m.values, 'b-')
    ax11.set_xlabel('Month')
    # Make the y-ax1is label, ticks and tick labels match the line color.
    ax11.set_ylabel('Mean FWI 2015', color='b')
    ax11.tick_params('y', colors='b')
    ax112 = ax11.twinx()
    ax112.set_ylabel(y2_label + ' 2015', color='r')
    ax112.bar(ba_m.index.values, ba_m, color='r', alpha=.6)
    ax112.tick_params('y', colors='r')
    ax11.set_title(list(bboxes.keys())[0])

    ax12 = plt.subplot2grid((2, 4), (1, 1), colspan=1)
    fwi_m = ds_monthly_means(spatial_subset(fwi15, bboxes['Kalimantan']), land_mask)
    ba_m = dfr_monthly_counts(spatial_subset_dfr(ba15, bboxes['Kalimantan']))
    print(bboxes['Kalimantan'])
    ax12.plot(months, fwi_m.values, 'b-')
    ax12.set_xlabel('Month')
    # Make the y-ax1is label, ticks and tick labels match the line color.
    ax12.set_ylabel('Mean FWI 2015', color='b')
    ax12.tick_params('y', colors='b')
    ax122 = ax12.twinx()
    ax122.bar(ba_m.index.values, ba_m, color='r', alpha=.6)
    ax122.set_ylabel(y2_label + ' 2015', color='r')
    ax122.tick_params('y', colors='r')
    ax12.set_title(list(bboxes.keys())[1])


    ax13 = plt.subplot2grid((2, 4), (1, 2), colspan=1)
    fwi_m = ds_monthly_means(spatial_subset(fwi15, bboxes['South Sumatra']), land_mask)
    ba_m = dfr_monthly_counts(spatial_subset_dfr(ba15, bboxes['South Sumatra']))
    print(bboxes['South Sumatra'])
    ax13.plot(months, fwi_m.values, 'b-')
    ax13.set_xlabel('Month')
    # Make the y-ax1is label, ticks and tick labels match the line color.
    ax13.set_ylabel('Mean FWI 2015', color='b')
    ax13.tick_params('y', colors='b')
    ax132 = ax13.twinx()
    ax132.bar(ba_m.index.values, ba_m, color='r', alpha=.6)
    ax132.set_ylabel(y2_label + ' 2015', color='r')
    ax132.tick_params('y', colors='r')
    ax13.set_title(list(bboxes.keys())[2])

    ax14 = plt.subplot2grid((2, 4), (1, 3), colspan=1)
    fwi_m = ds_monthly_means(spatial_subset(fwi15, bboxes['Inner Riau']), land_mask)
    ba_m = dfr_monthly_counts(spatial_subset_dfr(ba15, bboxes['Inner Riau']))
    ax14.plot(months, fwi_m.values, 'b-')
    ax14.set_xlabel('Month')
    # Make the y-ax1is label, ticks and tick labels match the line color.
    ax14.set_ylabel('Mean FWI 2015', color='b')
    ax14.tick_params('y', colors='b')
    ax142 = ax14.twinx()
    ax142.bar(ba_m.index.values, ba_m, color='r', alpha=.6)
    ax142.set_ylabel(y2_label + ' 2015', color='r')
    ax142.tick_params('y', colors='r')
    ax14.set_title(list(bboxes.keys())[3])


    fig.suptitle(suptitle, size=16)
    #fig.suptitle('MODIS FRP Collection 6', size=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(fig_name, res=300)
    plt.show()




#TODO
#plot 2008 - 2015 and 2015 for
#indonesia and amazon
#FWI and DC vs BA, FRP, GFED4, GFED4.1
"""
indonesia_bbox = [7.0, -11.0, 93.0, 143.0]
kalimantan = [7.0, -4.5, 108.0, 119]
sumatra_south = [3, -6, 98, 106]
riau_inner = [1,  -0.4, 101, 103.5]
bboxes = {'Indonesia': indonesia_bbox,
          'Kalimantan': kalimantan,
          'South Sumatra': sumatra_south,
          'Inner Riau': riau_inner}


land_mask = '~/data/land_mask/land_mask_indonesia.nc'
fwi_ds = '~/data/fwi/fwi_dc_indonesia.nc'
ba_prod = '~/data/ba/indonesia_ba.parquet'
frp_prod = '~/data/frp/M6_indonesia.parquet'

#orig_indonesia_bbox = [8.0, -13.0, 93.0, 143.0]
indonesia_bbox = [7.0, -11.0, 93.0, 143.0]
kalimantan = [7.0, -4.5, 108.0, 119]
sumatra_south = [3, -6, 98, 106]
riau_inner = [1,  -0.4, 101, 103.5]
bboxes = {'Indonesia': indonesia_bbox,
          'Kalimantan': kalimantan,
          'South Sumatra': sumatra_south,
          'Inner Riau': riau_inner}

land_mask = xr.open_dataset(land_mask)
ba = pd.read_parquet(ba_prod)
frp = pd.read_parquet(frp_prod)

fwi = xr.open_dataset('~/data/fwi/fwi_dc_indonesia.nc')
"""

#def plot_comp(fwi, ba, bboxes, land_mask, suptitle, y2_label, fig_name):
"""
plot_comp_gen(fwi['fwi'], ba, bboxes, land_mask, 'MCD64A1_BA', 'MCD64A1 ba pixel count', 'MCD64A1_indonesia.png')
plot_comp_gen(fwi, ba, bboxes, land_mask, 'FWI', 'MCD64A1 BA', 'MCD64A1 BA count', 'MCD64A1_BA_indonesia.png')
def plot_comp_gen(fwi, ba, bboxes, land_mask, ds_label, suptitle, y2_label, fig_name):
"""

