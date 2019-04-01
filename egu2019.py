import io
import pandas as pd
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import seaborn as sns
import matplotlib.pyplot as plt
from cartopy import feature
from matplotlib.colors import ListedColormap
import cartopy.io.shapereader as shapereader
from gridding import Gridder

colorl = [
'#14213d','#19223e','#1f233e','#23243f','#282540','#2d2640','#302741','#352842','#382942','#3c2a43','#402b44','#442c44','#472d45','#4c2e46','#4f2f46','#533047','#563148','#593248','#5d3349','#60344a','#64354a','#68364b','#6b374c','#70384c','#72394d','#763a4e','#7a3b4e','#7e3c4f','#823d4f','#853d50','#893f51','#8d3f51','#904052','#944153','#974253','#9b4354','#9f4455','#a34555','#a74656','#aa4756','#af4757','#b24858','#b64958','#ba4a59','#bd4b5a','#c14c5a','#c54d5b','#c94d5b','#ce4e5c','#d14f5d','#d5505d','#d9515e','#dc525f','#e0525f','#e55360','#e95461','#ec5561','#f05662','#f45762','#f85763','#fc5864','#ff5a64','#ff6064','#ff6563','#ff6a63','#ff6e63','#ff7362','#ff7662','#ff7b61','#ff7f61','#ff8461','#ff8860','#ff8b60','#ff8f5f','#ff925f','#ff965e','#ff9a5e','#ff9d5d','#ffa25d','#ffa55c','#ffa95c','#ffac5b','#ffb05a','#ffb25a','#ffb759','#ffba58','#ffbd58','#ffc057','#ffc456','#ffc755','#ffca55','#ffce54','#ffd053','#ffd452','#ffd751','#ffdb50','#ffdd4f','#ffe04e','#ffe34d','#ffe74c']
#colorl = [
#'#000000','#060202','#0d0505','#110807','#160a09','#190c0b','#1c0f0d','#1e100f','#221311','#241313','#271514','#2b1615','#2e1717','#321818','#341a19','#381b1a','#3c1c1c','#3f1d1d','#421f1f','#452020','#492121','#4c2222','#502324','#542425','#572526','#5b2728','#5f2829','#62292b','#662b2c','#692c2d','#6d2d2e','#712e30','#742f31','#783133','#7b3134','#7f3335','#843437','#873538','#8b363a','#8f383b','#93393c','#973a3e','#9b3b40','#9f3c41','#a33e42','#a63f44','#ab4045','#af4247','#b24348','#b6444a','#ba454b','#c0464d','#c3484e','#c7494f','#cc4a51','#cf4b53','#d44c54','#d84e56','#dc4f57','#e05059','#e5515a','#e9535c','#ed545d','#f1555f','#f65661','#f95762','#fd5963','#ff5e64','#ff6463','#ff6963','#ff6f63','#ff7362','#ff7962','#ff7d61','#ff8361','#ff8860','#ff8b60','#ff915f','#ff945f','#ff985e','#ff9e5d','#ffa15d','#ffa65c','#ffaa5b','#ffae5b','#ffb25a','#ffb559','#ffba58','#ffbd58','#ffc257','#ffc556','#ffca55','#ffcc54','#ffd053','#ffd452','#ffd851','#ffdc50','#ffdf4f','#ffe34d','#ffe74c'
#          ]
my_cmap = ListedColormap(sns.color_palette(colorl).as_hex())

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
                               facecolor='#14213D',
                               edgecolor='none',
                               alpha = 1)
    return ind_state_borders

def plot_ind_data(indon, dataset, variable, my_cmap, vmin, vmax):
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_axes([0, 0, 1, 1], projection = ccrs.PlateCarree())
    #ax.add_feature(feature.COASTLINE)
    ax.add_feature(indon, zorder=0)
    ax.set_extent((94.8, 141.1, 6.2, -10.5))
    ax.outline_patch.set(visible=False)
    pcm = ax.pcolormesh(ds.longitude - (gri.step / 2), ds.latitude - (gri.step / 2), ds[variable],
                 transform=ccrs.PlateCarree(), cmap=my_cmap, vmin=vmin, vmax=vmax)
    cax = fig.add_axes([0.1, 0.1, 0.2, 0.03])
    cb = plt.colorbar(pcm, cax = cax, orientation = 'horizontal', pad = 0)
    cb.ax.set_xlabel('Fire ocurence probability')
    ax.background_patch.set_alpha(0)
    ax.patch.set_alpha(0)
    plt.savefig("egu_poster/figures/data_{}.png".format(variable), format="png", dpi=300,
            bbox_inches='tight', pad_inches = 0, transparent = True)
    plt.show()

def get_probs(dfr, model, year, month):
    probs = dfr[[model, 'lonind', 'latind']][(dfr.year == year) & (dfr.month == month)]
    return probs


gri = Gridder(bbox = 'indonesia', step = 0.25)
dfr = pd.read_parquet('data/prob_dfr.parquet')
variable = 'frp'
probs = get_probs(dfr, variable, 2015, 10)
ds = gri.dfr_to_dataset(probs, variable, np.nan)
indon = ind_states()
plot_ind_data(indon, ds, variable, my_cmap, 1, 300)

