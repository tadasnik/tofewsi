import io
import matplotlib
matplotlib.font_manager._rebuild()
font = {'family' : 'Montserrat',
        'weight' : 'regular',
        'size'   : 20}
COLOR = '#1A2D40'
matplotlib.rcParams['text.color'] = COLOR
matplotlib.rcParams['axes.labelcolor'] = COLOR
matplotlib.rcParams['xtick.color'] = COLOR
matplotlib.rcParams['ytick.color'] = COLOR
matplotlib.rc('font', **font)

import pandas as pd
import xarray as xr
import numpy as np
from scipy import interp
import cartopy.crs as ccrs
import seaborn as sns
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from cartopy import feature
from matplotlib.colors import ListedColormap
import cartopy.io.shapereader as shapereader
from gridding import Gridder
import sklearn.metrics as skm

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

def plot_ind_data(dfr, indon, variable, my_cmap, vmin, vmax, year=None, month=None, log_scale=None):
    ds = get_variable_ds(dfr, variable, year, month)
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_axes([0, 0, 1, 1], projection = ccrs.PlateCarree())
    print(ds[variable].min(), ds[variable].max())
    #ax.add_feature(feature.COASTLINE)
    fc_c = my_cmap(0)

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

def do_roc_year(frpsel, features, clf, max_fact):
    tprs = []
    tprsint = []
    fprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for year in range(2002, 2019, 1):
        probas_, score, y_test = predict_year(frpsel, features, year, clf, max_fact)
        #if not score:
        #    continue
        if clf == 'maxent':
            fpr, tpr, thresholds = roc_curve(y_test, probas_)
        else:
            fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
        # Compute ROC curve and area the curve
        fprs.append(fpr)
        tprs.append(tpr)
        tprsint.append(interp(mean_fpr, fpr, tpr))

        tprsint[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        print('roc_auc', roc_auc)
        aucs.append(roc_auc)
    return tprs, tprsint, fprs, aucs, mean_fpr, score


def roc_plots(dfr, name, color):
    dfr['label'] = 0
    dfr['label'][dfr.frp > 10] = 1
    plot_x_size = 6 * 1.2
    plot_y_size = 26 * 1.2
    fig, axes = plt.subplots(nrows = 1, ncols=4,
                             figsize = (plot_y_size, plot_x_size))
    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.05, hspace=0.2, wspace=0.1)
    mod_names = {'Logistic_prob': 'Logistic regression', 'Maxent_prob': 'Maxent', 'SVC rbf_prob': 'Support Vector Classifier RBF kernel', 'NeuralNet_prob': 'Neural Networks'}
    mean_fpr = np.linspace(0, 1, 100)
    for nr, ax in enumerate(axes):
        column = list(mod_names.keys())[nr]
        mod = mod_names[column]
        sc = dfr.groupby(['year']).apply(lambda x: skm.roc_curve(x.label, x[column]))
        #tprs, tprsint, fprs, aucs, mean_fpr, score  = do_roc(X, XX, y, clf)
        tprsint = []
        aucs = []
        for item in sc.values:
            tprsint.append(interp(mean_fpr, item[0], item[1]))
            rauc = skm.auc(item[0], item[1])
            aucs.append(rauc)
            ax.plot(item[0], item[1], lw=1, color='#5D5D5D', alpha=0.3)#,
        tprsint[-1][0] = 0.0
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='#464646', alpha = .8)
        #         label='Chance', alpha=.8)
        mean_tpr = np.mean(tprsint, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = skm.auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color=color,
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=5)
        #colax.text(0.7, .2, 'score {0:.2}'.format(score))

        std_tpr = np.std(tprsint, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.patch.set_alpha(0)
        if nr == 0:
            ax.set_ylabel('True Positive Rate')
        ax.set_xlabel('False Positive Rate')
        ax.set_title(mod)
        ax.legend(loc="lower right",framealpha=0)
#fig.text(0.5, 0.01, 'Mean {}'.format(fwi_ds), ha='center', fontsize = 14)
    #fig.text(0.01, 0.5, 'Active fire pixel count', va='center', rotation='vertical', fontsize = 14)
    #tit = fig.suptitle('{0}'.format(name), y=.97, fontsize=18)
    plt.savefig("egu_poster/figures/rocs_{}.png".format(name), format="png", dpi=300,
            bbox_inches='tight', pad_inches = 0, transparent = True)

    #plt.savefig('figs/rocs_{}_indonesia.png'.format(name), dpi=300)#, bbox_inches='tight', bbox_extra_artists=[tit])
    #plt.savefig('egu_poster/figures/rocs_{}_indonesia.png'.format(name), dpi=300)#, bbox_inches='tight', bbox_extra_artists=[tit])
    plt.show()


def get_probs(dfr, model, year, month):
    probs = dfr[[model, 'lonind', 'latind']][(dfr.year == year) & (dfr.month == month)]
    return probs

def get_variable_ds(dfr, variable, year = None, month = None):
    if year:
        probs = get_probs(dfr, variable, year, month)
    else:
        probs = dfr
    ds = gri.dfr_to_dataset(probs, variable, np.nan)
    return ds

def plot_scores_bar_single(score1, score2, name):
    years = range(2002, 2019, 1)
    ind = np.arange(len(years))
    width = 0.35       # the width of the bars
    plot_x_size = 6 * 1.2
    plot_y_size = 22 * 1.2
    fig, ax = plt.subplots(figsize = (plot_y_size, plot_x_size))
    rects1 = ax.bar(ind, score1, width, color='#FF5964')
    rects2 = ax.bar(ind + width, score2, width, color='#35A7FF')
    ax.set_xticks(ind + width / 2)
    ax.set_yticks([.1,.3,.5,.7,.9])
    ax.set_xticklabels(years)
    ax.patch.set_alpha(0)
    ax.set_xlim(-0.4, 16.8)
    ax.spines['bottom'].set_position(('outward', 10))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.savefig("egu_poster/figures/scores_{}_single.png".format(name), format="png", dpi=300,
            bbox_inches='tight', pad_inches = 0, transparent = True)
    plt.show()


def plot_scores_bar(score1, score2, metric1, metric2, name):
    years = range(2002, 2019, 1)
    ind = np.arange(len(years))
    width = 0.35       # the width of the bars
    plot_x_size = 4 * 1.2
    plot_y_size = 36 * 1.2
    fig, ax = plt.subplots(nrows = 1, ncols=2,
                             figsize = (plot_y_size, plot_x_size))
    rects1 = ax[0].bar(ind, score1, width, color='#FF5964')
    rects2 = ax[0].bar(ind + width, score2, width, color='#FFE74C')
    ax[0].set_xticks(ind + width / 2)
    ax[0].set_xticklabels(years)
    ax[0].patch.set_alpha(0)

    rects1 = ax[1].bar(ind, metric1, width, color='#FF5964')
    rects2 = ax[1].bar(ind + width, metric2, width, color='#FFE74C')
    ax[1].set_xticks(ind + width / 2)
    ax[1].set_xticklabels(years)
    ax[1].patch.set_alpha(0)
    plt.savefig("egu_poster/figures/scores_{}.png".format(name), format="png", dpi=300,
            bbox_inches='tight', pad_inches = 0, transparent = True)
    plt.show()



def get_scores(dfr, grouper, column):
    dfr['label'] = 0
    dfr['label'][dfr.frp > 10] = 1
    frp = dfr.groupby([grouper])['frp'].sum()
    prob = dfr.groupby([grouper])['NeuralNet_prob'].sum()
    rocsc = dfr.groupby([grouper]).apply(lambda x: skm.roc_auc_score(x.label, x[column], average="weighted"))
    briersc = dfr.groupby([grouper]).apply(lambda x: skm.brier_score_loss(x.label, x[column]))
    logloss = dfr.groupby([grouper]).apply(lambda x: skm.log_loss(x.label, x[column]))
    prsc = dfr.groupby([grouper]).apply(lambda x: skm.average_precision_score(x.label, x[column]))
    recal = dfr.groupby([grouper]).apply(lambda x: skm.recall_score(x.label, x[column].round().astype(int)))
    return rocsc, briersc, logloss, prsc, recal

gri = Gridder(bbox = 'indonesia', step = 0.25)
dfr = pd.read_parquet('data/prob_dfr_v2.parquet')
dfrfwi = pd.read_parquet('data/prob_dfr_fwi_only.parquet')
indon = ind_states()
year = 2015
month = 10

rocsc, briersc, loglos, prsc, recal = get_scores(dfr, 'year', 'NeuralNet_prob')
rocscfwi, brierscfwi, loglos, prscfwi, recalfwi = get_scores(dfrfwi, 'year', 'NeuralNet_prob')

dfr['label'] = 0
dfr['label'][dfr.frp > 10] = 1
#brier = dfr.groupby(['lonind', 'latind']).apply(lambda x: skm.recall_score(x.label,
#                                                x['NeuralNet_prob'].round().astype(int)))

#brier = dfr.groupby(['lonind', 'latind']).apply(lambda x: skm.average_precision_score(x.label, x['NeuralNet_prob']))
#brier = brier.reset_index()
#brier.rename({0: 'briersc'}, axis=1, inplace=True)
#plot_ind_data(brier, indon, 'briersc', my_cmap, 0, 1)
#plot_scores_bar(briersc, brierscfwi, rocsc, rocscfwi, 'brier_roc')
#plot_scores_bar_single(briersc, rocsc, 'brier_roc')
plot_scores_bar_single(recal, prsc, 'recpr')
#plot_ind_data(dfr, indon, 'frp', my_cmap, 10, 1000, year, month, log_scale=True)
#plot_ind_data(dfr, indon, 'NeuralNet_prob', my_cmap, 0, 1, year, month)
#plot_ind_data(dfrfwi, indon, 'ffmc_med', my_cmap, 30, 90, year, month)
#plot_ind_data(dfr, indon, 'fwi_med', my_cmap, 0, 30, year, month)
#plot_ind_data(dfr, indon, 'dc_med', my_cmap, 0, 1000, year, month)
#plot_ind_data(dfr, indon, 'f_prim', my_cmap_blue, 0, 1, year, month)
#plot_ind_data(dfr, indon, 'loss_accum_prim', my_cmap_red, 0, 0.2, year, month )
#plot_ind_data(dfr, indon, 'gain',  my_cmap_blue, 0, 0.4, year = year, month = month)
#plot_ind_data(dfr, indon, 'peat_depth',  my_cmap, 0, 800, year = year, month = month)
#roc_plots(dfr, 'full', '#FF5964')
#roc_plots(dfrfwi, 'fwi', '#FFE74C')
