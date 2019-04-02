import io
import pandas as pd
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import seaborn as sns
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from cartopy import feature
from matplotlib.colors import ListedColormap
import cartopy.io.shapereader as shapereader
from gridding import Gridder

colorl = [
#'#e8e8e8','#e9e7e6','#e9e6e5','#eae5e4','#ebe3e2','#ebe1e1','#ece1e0','#ecdfde','#eddedd','#eedddc','#eedbda','#efd9d8','#efd8d7','#f0d7d5','#f0d5d4','#f1d4d3','#f1d3d2','#f2d2d0','#f2d0cf','#f3cfcd','#f3cdcc','#f4ccca','#f4cbc9','#f5cac7','#f5c8c6','#f5c8c5','#f6c6c3','#f6c4c2','#f7c3c0','#f7c2bf','#f7c1be','#f8bfbd','#f8bebb','#f8bdba','#f9bbb8','#f9b9b7','#f9b8b6','#fab7b4','#fab6b3','#fab5b2','#fab4b1','#fbb2af','#fbb1ae','#fbafad','#fcadab','#fcacaa','#fcaaa8','#fcaaa7','#fca8a6','#fda7a4','#fda6a3','#fda4a1','#fda3a0','#fda29f','#fea09d','#fe9f9c','#fe9d9b','#fe9b9a','#fe9a99','#fe9997','#ff9795','#ff9594','#ff9493','#ff9391','#ff9190','#ff908f','#ff8e8e','#ff8d8d','#ff8c8b','#ff8b8a','#ff8988','#ff8788','#ff8687','#ff8485','#ff8384','#ff8182','#ff8082','#ff7e80','#ff7d7f','#ff7b7d','#ff7a7c','#ff797b','#ff7679','#ff7579','#ff7377','#ff7276','#ff7175','#ff6e73','#ff6e72','#ff6c71','#ff696f','#ff676e','#ff676d','#ff656c','#ff626a','#ff6069','#ff5f68','#ff5d66','#ff5b65','#ff5964'
'#1a2d40','#343243','#483747','#5e3d4a','#73414e','#864651','#9b4a54','#b04e58','#c5515b','#da545e','#ef5762','#ff6164','#ff7962','#ff8d60','#ffa05d','#ffb25a','#ffc356','#ffd452','#ffe54d','#fff19d'

#'#1a2d40','#493847','#73424e','#9c4a55','#c7515b','#f25762','#ff7a62','#ffa25d','#ffc656','#ffe74c'
#'#1a2d40','#383344','#513948','#6a404c','#824550','#9a4a54','#b34e58','#cc525c','#e65660','#ff5964'
#'#14213d','#19223e','#1f233e','#23243f','#282540','#2d2640','#302741','#352842','#382942','#3c2a43','#402b44','#442c44','#472d45','#4c2e46','#4f2f46','#533047','#563148','#593248','#5d3349','#60344a','#64354a','#68364b','#6b374c','#70384c','#72394d','#763a4e','#7a3b4e','#7e3c4f','#823d4f','#853d50','#893f51','#8d3f51','#904052','#944153','#974253','#9b4354','#9f4455','#a34555','#a74656','#aa4756','#af4757','#b24858','#b64958','#ba4a59','#bd4b5a','#c14c5a','#c54d5b','#c94d5b','#ce4e5c','#d14f5d','#d5505d','#d9515e','#dc525f','#e0525f','#e55360','#e95461','#ec5561','#f05662','#f45762','#f85763','#fc5864','#ff5a64','#ff6064','#ff6563','#ff6a63','#ff6e63','#ff7362','#ff7662','#ff7b61','#ff7f61','#ff8461','#ff8860','#ff8b60','#ff8f5f','#ff925f','#ff965e','#ff9a5e','#ff9d5d','#ffa25d','#ffa55c','#ffa95c','#ffac5b','#ffb05a','#ffb25a','#ffb759','#ffba58','#ffbd58','#ffc057','#ffc456','#ffc755','#ffca55','#ffce54','#ffd053','#ffd452','#ffd751','#ffdb50','#ffdd4f','#ffe04e','#ffe34d','#ffe74c'
]
colorlb = [
'#ffffff','#f7faff','#eff6ff','#e7f0ff','#dfebff','#d7e7ff','#cfe2ff','#c6deff','#bcd9ff','#b5d4ff','#abd0ff','#a1cbff','#98c6ff','#8dc2ff','#83bdff','#76b8ff','#6bb4ff','#5bb0ff','#4babff','#35a7ff'
#'#e8e8e8','#e2e5e9','#dae1eb','#d4deec','#cddaed','#c6d6ef','#bed2f0','#b8cff1','#b0ccf2','#a9c9f4','#a0c5f5','#98c1f6','#8fbef7','#85baf8','#7cb8f9','#72b4fb','#65b1fc','#59aefd','#4baafe','#35a7ff'
]
colorlr = [
'#ffffff','#fff8f7','#fff0ee','#ffe8e7','#ffe1df','#ffd9d7','#ffd2cf','#ffcac7','#ffc2bf','#ffb9b7','#ffb1af','#ffa9a6','#ffa19f','#ff9896','#ff8f8e','#ff8485','#ff7b7d','#ff7075','#ff656d','#ff5964'
]
#colorl = [
#'#000000','#060202','#0d0505','#110807','#160a09','#190c0b','#1c0f0d','#1e100f','#221311','#241313','#271514','#2b1615','#2e1717','#321818','#341a19','#381b1a','#3c1c1c','#3f1d1d','#421f1f','#452020','#492121','#4c2222','#502324','#542425','#572526','#5b2728','#5f2829','#62292b','#662b2c','#692c2d','#6d2d2e','#712e30','#742f31','#783133','#7b3134','#7f3335','#843437','#873538','#8b363a','#8f383b','#93393c','#973a3e','#9b3b40','#9f3c41','#a33e42','#a63f44','#ab4045','#af4247','#b24348','#b6444a','#ba454b','#c0464d','#c3484e','#c7494f','#cc4a51','#cf4b53','#d44c54','#d84e56','#dc4f57','#e05059','#e5515a','#e9535c','#ed545d','#f1555f','#f65661','#f95762','#fd5963','#ff5e64','#ff6463','#ff6963','#ff6f63','#ff7362','#ff7962','#ff7d61','#ff8361','#ff8860','#ff8b60','#ff915f','#ff945f','#ff985e','#ff9e5d','#ffa15d','#ffa65c','#ffaa5b','#ffae5b','#ffb25a','#ffb559','#ffba58','#ffbd58','#ffc257','#ffc556','#ffca55','#ffcc54','#ffd053','#ffd452','#ffd851','#ffdc50','#ffdf4f','#ffe34d','#ffe74c'
#          ]
my_cmap = ListedColormap(sns.color_palette(colorl).as_hex())
my_cmap_blue = ListedColormap(sns.color_palette(colorlb).as_hex())
my_cmap_red = ListedColormap(sns.color_palette(colorlr).as_hex())

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
                               facecolor='#1A2D40',
                               edgecolor='none',
                               alpha = 1)
    return ind_state_borders

def plot_ind_data(dfr, indon, variable, year, month, my_cmap, cmap_title, vmin, vmax, log_scale=None):
    ds = get_variable_ds(dfr, year, month, variable)
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_axes([0, 0, 1, 1], projection = ccrs.PlateCarree())
    print(ds[variable].min(), ds[variable].max())
    #ax.add_feature(feature.COASTLINE)
    fc_c = my_cmap(vmin)
    ax.add_feature(indon, zorder=0, facecolor=fc_c)
    ax.set_extent((94.8, 141.1, 6.2, -10.5))
    ax.outline_patch.set(visible=False)
    if log_scale:
        my_cmap.set_bad(('none'))
        pcm = ax.pcolormesh(ds.longitude - (gri.step / 2), ds.latitude - (gri.step / 2),
                            ds[variable], transform=ccrs.PlateCarree(),
                            norm = colors.LogNorm(vmin = vmin, vmax = ds[variable].max()),
                            cmap=my_cmap)
    else:
        pcm = ax.pcolormesh(ds.longitude - (gri.step / 2), ds.latitude - (gri.step / 2),
                            ds[variable], transform=ccrs.PlateCarree(),
                            cmap=my_cmap, vmin=vmin, vmax=vmax)
    cax = fig.add_axes([0.07, 0.12, 0.2, 0.03])
    cb = plt.colorbar(pcm, cax = cax, orientation = 'horizontal',  pad = 0)
    #cb.ax.set_title(cmap_title)
    ax.background_patch.set_alpha(0)
    ax.patch.set_alpha(0)
    plt.savefig("egu_poster/figures/{0}_{1}_{2}.png".format(variable, year, month), format="png", dpi=300,
            bbox_inches='tight', pad_inches = 0, transparent = True)
    plt.show()

def get_probs(dfr, model, year, month):
    probs = dfr[[model, 'lonind', 'latind']][(dfr.year == year) & (dfr.month == month)]
    return probs

def get_variable_ds(dfr, year, month, variable):
    probs = get_probs(dfr, variable, year, month)
    ds = gri.dfr_to_dataset(probs, variable, np.nan)
    return ds




gri = Gridder(bbox = 'indonesia', step = 0.25)
dfr = pd.read_parquet('data/prob_dfr.parquet')
dfrfwi = pd.read_parquet('data/prob_dfr_fwi_only.parquet')
indon = ind_states()
year = 2015
month = 10

#plot_ind_data(dfr, indon, 'frp', year, month, my_cmap, 10, 1, log_scale=True)
#plot_ind_data(dfrfwi, indon, 'NeuralNet_prob', year, month, my_cmap, 'None', 0, 1)
plot_ind_data(dfrfwi, indon, 'ffmc_med', year, month, my_cmap, 'None', 30, 90)
#plot_ind_data(dfr, indon, 'f_prim', year, month, my_cmap_blue, 'Fraction primary forest', 0, 1)
#plot_ind_data(dfr, indon, 'loss_accum_prim', year, month, my_cmap_red, 'Fraction primary forest', 0, 0.2)
