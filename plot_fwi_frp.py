import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from fwi_fire import CompData

SEA_lulc = {0: 'No class',
            1: 'Water',
            2: 'Mangrove',
            3: 'Peatswamp forest',
            4: 'Lowland forest',
            5: 'Lower montane forest',
            6: 'Upper montane forest',
            7: 'Regrowth/plantation',
            8: 'Lowland mosaic',
            9: 'Montane mosaic',
            10: 'Lowland open',
            11: 'Lower montane forest',
            12: 'Urban',
            13: 'Large-scale palm'}

def violin_plot():
    df = pd.read_pickle('lulc_frp')
    df.boxplot('frp', by = 'lulc')
    plt.title('Active fire (FRP) counts per grid cell grouped per LULC' )
    plt.suptitle("")
    plt.ylabel('FRP count')
    plt.xticks(ticks = list(range(1,14,1)), labels = list(SEA_lulc.values())[1:], rotation = 'vertical')
    plt.tight_layout()
    plt.show()

def region_scatter(cc, lats, lons, name, fwi_ds):
    plot_y_size = len(lats) * 3
    plot_x_size = len(lons) * 3
    fig, axes = plt.subplots(nrows=len(lats), ncols=len(lons),
                             figsize = (plot_y_size, plot_x_size),
                             sharex = True, sharey = True)
    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.05, hspace=0.2, wspace=0.1)
    for row_nr, rowax in enumerate(axes):
        for col_nr, colax in enumerate(rowax):
            print(row_nr, col_nr)
            frp_pix, fwi_pix, lulc = cc.get_pixel(lats[row_nr], lons[col_nr], fwi_ds)
            print(lulc)
            try:
                lulc = SEA_lulc[lulc.values[0]]
            except:
                lulc = SEA_lulc[0]
            colax.scatter(fwi_pix, frp_pix)
            #colax.set_title('lat {0} lon {1}'.format(lats[row_nr], lons[col_nr]))
            colax.set_title(lulc)
    #plt.tight_layout()
    fig.text(0.5, 0.01, 'Mean {}'.format(fwi_ds), ha='center', fontsize = 14)
    fig.text(0.01, 0.5, 'Active fire pixel count', va='center', rotation='vertical', fontsize = 14)
    tit = fig.suptitle('{0}'.format(name), y=.97, fontsize=16)
    plt.savefig('figures/active_fire_vs_{0}_{1}_monthly_lulc.png'.format(fwi_ds, region), dpi=80)#, bbox_inches='tight', bbox_extra_artists=[tit])

def lat_lon_grid_points(bbox, step):
    """
    Returns two lists with latitude and longitude grid cell center coordinates
    given the bbox and step.
    """
    lat_bbox = [bbox[0], bbox[2]]
    lon_bbox = [bbox[1], bbox[3]]
    latmin = lat_bbox[np.argmin(lat_bbox)]
    latmax = lat_bbox[np.argmax(lat_bbox)]
    lonmin = lon_bbox[np.argmin(lon_bbox)]
    lonmax = lon_bbox[np.argmax(lon_bbox)]
    numlat = int((latmax - latmin) / step) + 1
    numlon = int((lonmax - lonmin) / step) + 1
    lats = np.linspace(latmin, latmax, numlat, endpoint = True)
    lons = np.linspace(lonmin, lonmax, numlon, endpoint = True)
    return lats, lons

def region_means(cc, regions):
    region_names = list(regions.keys())
    plot_y_size = 10
    plot_x_size = 18
    fig, axes = plt.subplots(nrows=3, ncols=1,
                             figsize = (plot_x_size, plot_y_size),
                             sharex = True, sharey = True)
    #fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, hspace=0.2, wspace=0.)
    for row_nr, rowax in enumerate(axes):
        bbox = regions[region_names[row_nr]]
        reg_fwi = cc.spatial_subset(cc.fwi_m, bbox)
        reg_frp = cc.spatial_subset(cc.frp_m, bbox)
        reg_fwi_m = reg_fwi.mean(dim=['latitude', 'longitude']).to_dataframe()
        reg_frp_m = reg_frp.mean(dim=['latitude', 'longitude']).to_dataframe()
        dfr = pd.concat([reg_fwi_m, reg_frp_m], axis=1)
        dfr.rename({'count': 'frp count'}, inplace = True)
        dfr.plot(ax = rowax, secondary_y = ['fwi','count'], sharey = True)
        rowax.right_ax.set_ylim(0,200)
        rowax.set_title(region_names[row_nr])
        #colax.set_title('lat {0} lon {1}'.format(lats[row_nr], lons[col_nr]))
    plt.tight_layout()
    plt.savefig('figs/fwi_vs_frp_count_regional_mean_monthly.png',
                dpi=80, bbox_inches='tight')
    plt.show()

def get_pure_cell_values(cc, threshold, item, fwi_ds):
    ds = cc.lulc[str(item)]
    perc = ds / cc.lulc['total']
    mask = ds.where(perc > threshold, drop = True)
    dfr = mask.to_dataframe(name = item)
    dfr.dropna(inplace = True)
    dfr.reset_index(inplace = True)
    frps = []
    fwis = []
    for nr, row in dfr.iterrows():
        frp_pix, fwi_pix = cc.get_pixel(row['latitude'], row['longitude'], fwi_ds)
        frps.extend(frp_pix.values)
        fwis.extend(fwi_pix.values)
    return frps, fwis



data_path = '/mnt/data/'
cc = CompData(data_path)
cc.read_lulc()
cc.read_monthly()


"""
regions = {'South_East_Sumatra': [-3, 103, -4, 104],
           'Peatland_east_Riau': [1, 101.5, 0, 102.5],
           'South_Kalimantan': [-2.25, 112, -3.25, 113]}

for region in regions.keys():
    lats, lons = lat_lon_grid_points(regions[region], 0.25)
    for fwi_ds in ['fwi', 'dc']:#, 'ffmc']:
        region_scatter(cc, lats, lons, region, fwi_ds)
"""

