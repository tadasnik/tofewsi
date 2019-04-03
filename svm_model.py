from __future__ import print_function
import geopandas as gpd
import codecs, json
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1 import AxesGrid
from gridding import Gridder
import numpy as np
import pandas as pd
from scipy import interp
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc, accuracy_score, roc_auc_score, balanced_accuracy_score, average_precision_score, precision_recall_curve, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
import rpy2.robjects as robj
from rpy2.robjects.lib import grid
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

def dfr_to_json(dfrs):
    gri = Gridder(bbox = 'indonesia', step = 0.25)
    year = 2002
    month_names = pd.date_range(start='2002-01'.format(year), freq='M', periods=12).month_name()
    years_d = {}
    for year in range(2002, 2019, 1):
        dfr = dfrs[dfrs.year == year]
        month_inds = dfr.month.unique()
        prob_cols = [col for col in dfr.columns if 'prob' in col]
        #dfs[prob_cols][dfs[prob_cols] < 0.5] = np.nan
        cols = prob_cols + ['longitude', 'latitude', 'frp']
        gri = Gridder(bbox = 'indonesia', step = 0.25)
        dfr = gri.spatial_subset_ind_dfr(dfr, gri.bbox)
        json_d = {}
        for nr, month in enumerate(month_names):
            print(year, nr)
            dfs = dfr[dfr.month == month_inds[nr]]
            json_d[month] = {
                    'frp': dfs.frp.astype(int).tolist(),
                    'dc': dfs.dc_med.astype(int).tolist(),
                    'fwi': dfs.fwi_med.astype(int).tolist(),
                    'ffmc': dfs.ffmc_med.astype(int).tolist(),
                    'Logistic': (dfs.Logistic_prob * 100).astype(int).tolist(),
                    'NN': (dfs.NeuralNet_prob * 100).astype(int).tolist(),
                    'Maxent': (dfs.Maxent_prob * 100).astype(int).tolist(),
                    'SVC': (dfs['SVC rbf_prob'] * 100).astype(int).tolist()
                    }
            if (year == 2002) & (nr == 0):
                dfs['latitude'] = gri.lat_bins[dfs.latind.values + 1]
                dfs['longitude'] = gri.lon_bins[dfs.lonind.values]
                dfs[['longitude', 'latitude']].to_json('/home/tadas/tofewsi/website/assets/geo/lonlats_all.json', orient="values")


        years_d[year] = json_d
    with open('/home/tadas/tofewsi/website/assets/probdata_all.json', 'w') as outfile:
        json.dump(years_d, outfile)




    """
    dfs = dfs[cols]
    dffi = dfs * 100
    dffi = dffi.astype('int')
    frpd = {'frp': dffi['frp'].tolist()}
    probd = {'svc_rbf': dffi['SVC rbf_prob'].tolist(),
            'NN': dffi['NeuralNet_prob'].tolist(),
            'Maxent': dffi['Maxent_prob'].tolist()}
    json.dump(topolonlatd, codecs.open('data/topolonlats.json', 'w', encoding='utf-8'))
    json.dump(frpd, codecs.open('data/frp.json', 'w', encoding='utf-8'))
    json.dump(probd, codecs.open('data/probs.json', 'w', encoding='utf-8'))
    """

def plot_year_probs(dfr, clfs, year):
    months = 12
    plot_y_size = months * 3
    plot_x_size = (len(clfs) + 1) * 6
    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes,
                  dict(map_projection = projection))
    fig = plt.figure(figsize=(plot_x_size, plot_y_size))
    axgr = AxesGrid(fig, 111, axes_class = axes_class,
                    nrows_ncols=(months, len(clfs) + 1),
                    axes_pad=.3,
                    cbar_mode='edge',
                    cbar_location='bottom',
                    cbar_pad=0.5,
                    cbar_size='3%',
                    label_mode='')

    #fig, axes = plt.subplots(nr ows = months, ncols = len(clfs) + 1,
    #                         figsize = (plot_y_size, plot_x_size),
    #                         sharex = True, sharey = True,
    #                         subplot_kw={'projection': ccrs.PlateCarree()})
    #fig.subplots_adjust(left=0.05, right=0.97, bottom=0.05, hspace=0.2, wspace=0.1)
    mod_names = list(clfs.keys())
    months = dfr.month.unique()
    gri = Gridder(bbox = 'indonesia', step = 0.25)
    for row_nr, rowax in enumerate(axgr.axes_row):
        for col_nr, colax in enumerate(rowax):
            df = dfr[dfr.month == months[row_nr]]
            try:
                col_name = mod_names[col_nr] + '_prob'
                vmin = 0.49
                vmax = 0.5
            except:
                col_name = 'frp'
                vmin = 9
                vmax = 10
            ds = gri.dfr_to_dataset(df, col_name, np.nan)
            print(col_name)
            print(ds[col_name])
            ds[col_name].plot.pcolormesh(ax=colax, transform=ccrs.PlateCarree(),
                                         vmin = vmin, vmax = vmax, x = 'longitude', y='latitude',
                                         add_colorbar=False, add_labels=False)
            #gl = colax.gridlines(ccrs.PlateCarree(), draw_labels=True)
            #gl.xlabels_top = False
            #colax.coastlines('50m')
            #colax.set_extent([9, 96, -10, 143], crs=ccrs.PlateCarree())
    tit = fig.suptitle('{0}'.format(year), y=.97, fontsize=18)
    plt.savefig('figs/models_probs_{}.png'.format(year), dpi=300)#, bbox_inches='tight', bbox_extra_artists=[tit])
    #plt.show()

def roc_plots(frpsel, features, clfs, cv, name, max_fact):
    plot_x_size = len(features) * 4
    plot_y_size = len(clfs) * 4
    fig, axes = plt.subplots(nrows=len(features), ncols=len(clfs),
                             figsize = (plot_y_size, plot_x_size),
                             sharex = True, sharey = True)
    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.05, hspace=0.2, wspace=0.1)
    mod_names = list(clfs.keys())
    for row_nr, rowax in enumerate(axes):
        for col_nr, colax in enumerate(rowax):
            #X = features[row_nr]
            #XX = feats_unscaled[row_nr]
            clf = clfs[mod_names[col_nr]]
            print(clf)
            tprs, tprsint, fprs, aucs, mean_fpr, score  = do_roc_year(frpsel, features[row_nr], clf, max_fact)
            #tprs, tprsint, fprs, aucs, mean_fpr, score  = do_roc(X, XX, y, clf)
            for item in zip(fprs, tprs):
               colax.plot(item[0], item[1], lw=1, color='#5D5D5D', alpha=0.3)#,
            #         label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

            #i += 1
            colax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='#464646', alpha = .8)
            #         label='Chance', alpha=.8)
            mean_tpr = np.mean(tprsint, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            colax.plot(mean_fpr, mean_tpr, color='#1A2D40',
                     label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                     lw=2, alpha=.8)
            #colax.text(0.7, .2, 'score {0:.2}'.format(score))

            std_tpr = np.std(tprsint, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            colax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                             label=r'$\pm$ 1 std. dev.')

            colax.set_xlim([-0.05, 1.05])
            colax.set_ylim([-0.05, 1.05])
            if row_nr == 0:
                colax.set_xlabel('False Positive Rate')
            if col_nr == 0:
                colax.set_ylabel('True Positive Rate')
            if row_nr == 0:
                colax.set_title(mod_names[col_nr])
            colax.legend(loc="lower right")
    #fig.text(0.5, 0.01, 'Mean {}'.format(fwi_ds), ha='center', fontsize = 14)
    #fig.text(0.01, 0.5, 'Active fire pixel count', va='center', rotation='vertical', fontsize = 14)
    tit = fig.suptitle('{0}'.format(name), y=.97, fontsize=18)
    #plt.savefig('figs/rocs_{}_indonesia.png'.format(name), dpi=300)#, bbox_inches='tight', bbox_extra_artists=[tit])
    plt.savefig('egu_poster/figures/rocs_{}_indonesia.png'.format(name), dpi=300)#, bbox_inches='tight', bbox_extra_artists=[tit])
    plt.show()

def fit_predict_maxent(x_train, y_train, x_test):
    robj.pandas2ri.activate()
    maxent = importr('maxnet')
    y_train_r = pandas2ri.py2ri(pd.Series(y_train))
    train_r = pandas2ri.py2ri(x_train)
    test_r = pandas2ri.py2ri(x_test)
    mod = maxent.maxnet(y_train_r, train_r)
    probs = robj.r('predict')(mod, test_r, type="logistic")
    probs = np.array(probs).flatten()
    return probs, None

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


def do_roc(X, XX, y, clf):
    tprs = []
    tprsint = []
    fprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for train, test in cv.split(X, y):
        if clf == 'maxent':
            print(train.shape)
            #try
            probas_ = fit_predict_maxent(XX.iloc[train, :], y[train], XX.iloc[test, :])
            score = 0.9999
            fpr, tpr, thresholds = roc_curve(y[test], probas_)
            print(fpr, tpr, thresholds)
            #except:
            #    pass
        else:
            probas_ = clf.fit(X[train], y[train]).predict_proba(X[test])
            score = clf.score(X[test], y[test])
            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        # Compute ROC curve and area the curve
        fprs.append(fpr)
        tprs.append(tpr)
        tprsint.append(interp(mean_fpr, fpr, tpr))
        tprsint[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
    return tprs, tprsint, fprs, aucs, mean_fpr, score

def class_labels(frpsel, threshold):
    labels = np.zeros_like(frpsel['frp'].values)
    labels[frpsel['frp'] > threshold] = 1
    return labels

def feature_selection(frpsel, max_fact):
    frpsel = balance_classes(frpsel)
    factor = subset_factor(frpsel.shape[0], max_fact)
    frpsel = frpsel.iloc[::factor, :]
    labels = class_labels(frpsel, 10)
    custom_cv  = leave_year_split(frpsel)
    features = frpsel.columns[4:,].tolist()
    feats =  features + ['lonind', 'latind']
    print('features', feats)
    XS = frpsel[feats]
    X_scaled = preprocessing.scale(XS.values)
    rfe = RFE(estimator=svmlin, n_features_to_select=1, step=1)
    rfe.fit(X_scaled, labels)
    print('feature ranking RFE: ', list(zip(feats, rfe.ranking_)))
    print("Optimal number of features : %d" % rfe.n_features_)
    n_estimators = 20
    clr = BaggingClassifier(svmlin, max_samples = 1.0 / n_estimators,
                            n_estimators=n_estimators, n_jobs=7)
    scores = cross_val_score(clr, X_scaled, labels, cv=custom_cv)
    print(scores)

    custom_cv  = leave_year_split(frpsel)
    rfecv = RFECV(estimator=svmlin, step=1, cv=custom_cv,
                  scoring='accuracy', n_jobs=7)
    #rfecv = RFECV(estimator=svmlin, step=1, cv=StratifiedKFold(9, shuffle = True),
    #              scoring='accuracy', n_jobs=7)
    print(labels)
    rfecv.fit(X_scaled, labels)
    print('feature ranking RFECV: ', list(zip(feats, rfecv.ranking_)))
    print("Optimal number of features : %d" % rfe.n_features_)
    print("Optimal number of features : %d" % rfecv.n_features_)
    print(rfecv.grid_scores_)
    return list(zip(feats, rfecv.ranking_))

def do_roc_auc(bboxes, clfs, max_fact):
    for key, item in bboxes.items():
        frpsel = frp_data_subset(item)
        #feats = [frpsel[['fwi', 'dc', 'ffmc']],
        #         frpsel[['loss_this', 'loss_last', 'loss_three', 'loss_accum', 'f_prim', 'gain']],
        feats =  [ ['fwi'], ['dc'], ['ffmc'], ['dc', 'ffmc'],
            ['fwi', 'dc', 'ffmc']]

        #feats =  [['fwi', 'dc', 'ffmc'], ['loss_this', 'loss_last', 'loss_three', 'loss_accum', 'f_prim', 'gain'], ['lonind', 'latind', 'loss_this', 'loss_last', 'loss_three', 'loss_accum', 'gain', 'fwi', 'ffmc'],
        #         ['lonind', 'latind', 'loss_this', 'loss_last', 'loss_three', 'loss_accum', 'f_prim', 'gain', 'fwi', 'dc', 'ffmc']]
        roc_plots(frpsel, feats, clfs, cv, key, max_fact)

def get_year_train_test(frpsel, year, max_fact=None):
    years = (((frpsel.month.astype(int) - 1) // 12.) + 2002).astype(int)
    X_train_inds = np.where(years != year)[0]
    X_test_inds = np.where(years == year)[0]
    if max_fact:
        factor = subset_factor(X_train_inds.shape[0], max_fact)
        X_train_inds = X_train_inds[::factor]
    return X_train_inds, X_test_inds

def predict_probability(x_train, y_train, x_test, y_test, clf):
    clf.fit(x_train, y_train)
    probas = clf.predict_proba(x_test)
    return probas, clf.score(x_test, y_test)

def frp_data_subset():
    #frp = pd.read_parquet('data/feature_frame_0.25deg_v2.parquet')
    frp = pd.read_parquet('data/feature_frame_0.25deg_v2_no_volcanoes.parquet')
    gri = Gridder(bbox = 'indonesia', step = 0.25)
    frp = gri.spatial_subset_ind_dfr(frp, gri.bbox)
    return frp

def balance_classes(frp):
    #make sample sizes evenish
    frp1 = frp[frp.frp > 10]
    frp0 = frp[frp.frp == 0]
    factor = int(frp0.shape[0] / (frp1.shape[0] * 2))
    frpsel = pd.concat([frp1, frp0.iloc[::factor, :]])
    return frpsel

def subset_factor(data_size, size_factor):
    factor = int(data_size // size_factor)
    print(factor)
    if factor == 0:
        factor = 1
    return factor

def select_nn_params(frpsel, max_fact):
    #features = ['fwi', 'dc', 'ffmc']
    #features = ['lonind', 'latind', 'loss_this', 'loss_last', 'loss_three', 'loss_accum', 'f_prim', 'gain', 'fwi', 'dc', 'ffmc']
    frpsel = balance_classes(frpsel)
    factor = subset_factor(frpsel.shape[0], max_fact)
    frpsel = frpsel.iloc[::factor, :]
    features = frpsel.columns[4:,].tolist()
    feats =  features + ['lonind', 'latind']
    scaler = preprocessing.StandardScaler().fit(frpsel[feats])
    labels = class_labels(frpsel, 10)
    feats = frpsel.loc[:, feats]
    feats = scaler.transform(feats)
    X_train, X_test, y_train, y_test = train_test_split(
        feats, labels, test_size=0.3, random_state=0)
    #custom_cv  = leave_year_split(frpsel)

# Set the parameters by cross-validation
    tuned_parameters = {'solver': ['lbfgs'], 'activation': ['logistic'],
                         'alpha': [1],
                        'hidden_layer_sizes': [(3, 6), (10), (2, 5, 10), (10, 5), (5, 10, 15)]}

    scores = ['precision', 'recall']

    for score in scores:
        custom_cv  = leave_year_split(frpsel)
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(MLPClassifier(), tuned_parameters, cv=custom_cv,
                           scoring='%s_macro' % score, n_jobs=-1)
        clf.fit(feats, labels)
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()


def select_svm_params(frpsel, max_fact):
    #features = ['fwi', 'dc', 'ffmc']
    #features = ['lonind', 'latind', 'loss_this', 'loss_last', 'loss_three', 'loss_accum', 'f_prim', 'gain', 'fwi', 'dc', 'ffmc']
    frpsel = balance_classes(frpsel)
    factor = subset_factor(frpsel.shape[0], max_fact)
    frpsel = frpsel.iloc[::factor, :]
    features = frpsel.columns[4:,].tolist()
    feats =  features + ['lonind', 'latind']
    scaler = preprocessing.StandardScaler().fit(frpsel[feats])
    labels = class_labels(frpsel, 10)
    feats = frpsel.loc[:, feats]
    feats = scaler.transform(feats)
    #X_train, X_test, y_train, y_test = train_test_split(
    #    feats, labels, test_size=0.3, random_state=0)

# Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.01, .05, 1e-1, 0.15, 0.18, 2e-1],
                         'C': [1, 10, 100]},
                        {'kernel': ['linear'], 'C': [1, 10, 100]}]

    scores = ['precision', 'recall']

    for score in scores:
        custom_cv  = leave_year_split(frpsel)
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(SVC(), tuned_parameters, cv=custom_cv,
                           scoring='%s_macro' % score, n_jobs=-1)
        #clf.fit(X_train, y_train)
        clf.fit(feats, labels)
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = labels, clf.predict(feats)
        print(classification_report(y_true, y_pred))
        print()

def leave_year_split(frpsel):
    year = 2002
    while year <= 2018:
        train, test = get_year_train_test(frpsel, year)#, 500)
        yield train, test
        year += 1


def predict_year(frpsel, features, year, clf, max_fact):
    train, test = get_year_train_test(frpsel, year)
    #features = ['fwi', 'dc', 'ffmc']
    #features = ['lonind', 'latind', 'loss_this', 'loss_last', 'loss_three', 'loss_accum', 'f_prim', 'gain', 'fwi', 'dc', 'ffmc']
    #scaler = preprocessing.StandardScaler().fit(frpsel[features])
    x_train = balance_classes(frpsel.iloc[train, :])
    y_train = class_labels(x_train, 10)
    factor = subset_factor(x_train.shape[0], max_fact)
    x_train = x_train.iloc[::factor, :]
    y_train = y_train[::factor]
    scaler = preprocessing.StandardScaler().fit(x_train[features])
    x_train_scaled = scaler.transform(x_train[features].values)
    x_test = frpsel.iloc[test, :]
    y_test = class_labels(x_test, 10)
    x_test_scaled = scaler.transform(x_test[features].values)
    print(year)
    print(x_train.shape)
    if clf == 'maxent':
        try:
            preds, score = fit_predict_maxent(x_train.loc[:, features], y_train, x_test.loc[:, features])
            print('predict_probability', score)
        except:
            return None, None, None
    else:

        preds, score = predict_probability(x_train_scaled, y_train, x_test_scaled, y_test, clf)
        print('predict_probability', score)
    return preds, score, y_test


def year_pred_to_dfr(year, max_fact, clfs, frpsel, features):
    train, test = get_year_train_test(frpsel, year)
    #features = ['fwi', 'dc', 'ffmc']
    #features = ['lonind', 'latind', 'loss_this', 'loss_last', 'loss_three', 'loss_accum', 'f_prim', 'gain', 'fwi', 'dc', 'ffmc']
    #scaler = preprocessing.StandardScaler().fit(frpsel[features])
    x_train = balance_classes(frpsel.iloc[train, :])
    y_train = class_labels(x_train, 10)
    factor = subset_factor(x_train.shape[0], max_fact)
    x_train = x_train.iloc[::factor, :]
    y_train = y_train[::factor]
    scaler = preprocessing.StandardScaler().fit(x_train[features])
    x_train_scaled = scaler.transform(x_train[features].values)
    x_test = frpsel.iloc[test, :]
    y_test = class_labels(x_test, 10)
    x_test_scaled = scaler.transform(x_test[features].values)
    for key, item in clfs.items():
        print(key)
        if key == 'Maxent':
            try:
                preds, score = fit_predict_maxent(x_train.loc[:, features], y_train, x_test.loc[:, features])
                x_test.loc[:, key + '_prob'] = preds
            except:
                return None, None, None
        else:
            preds, score = predict_probability(x_train_scaled, y_train, x_test_scaled, y_test, item)
            x_test.loc[:, key + '_prob'] = preds[:, 1]
    return x_test

def predict_years(clsf, frpsel, features, max_fact):
    dfrs = []
    for year in range(2002, 2019, 1):
        dfr = year_pred_to_dfr(year, max_fact, clfs, frpsel, features)
        print(dfr, year)
        dfr['year'] = year
        dfr['month'] = dfr['month'].astype(int)
        dfr['month'] = (dfr['month'] - dfr['month'].min()) + 1
        dfrs.append(dfr)
    dfrs = pd.concat(dfrs)
    return dfrs

sumatra = [6, 96, -6, 106]
java = [-5.8, 105, -9.3, 119]
sumatra = [6, 96, -6, 106]
indonesia = [9, 96, -10, 143]
kalimantan = [3, 108.7, -4.2, 119]
papua = [0, 130, -9.26, 141.5]
sulewasi = [1.5, 118.8, -5.6, 125.3]
riau = [3, 99, -2, 104]

"""
bboxes = {'Java_Bali': java,
          'Indonesia': indonesia,
          'Sumatra': sumatra,
          'Kalimantan': kalimantan,
          'Papua': papua,
          'Sulewasi': sulewasi,
          'Riau': riau}
"""

bboxes = {'Indonesia': indonesia}#,
         # 'Kalimantan': kalimantan,
         # 'Riau': riau}

random_state = np.random.RandomState(0)

cv = StratifiedKFold(n_splits=10, shuffle = True)

svmone = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.5)

svmlin = svm.SVC(kernel='linear', probability=True, C=1,
                     random_state=random_state)

svmrbf = svm.SVC(kernel='rbf', C=1, gamma=0.15, probability=True,
                     random_state=random_state)

logist = LogisticRegression(solver = 'liblinear', penalty='l1')

clfnn1 = MLPClassifier(solver='lbfgs', alpha=1,
                     hidden_layer_sizes=(10), activation='logistic', random_state=1)

clfnn2 = MLPClassifier(solver='lbfgs', alpha=2,
                     hidden_layer_sizes=(10), activation='logistic', random_state=1)


clfnn = MLPClassifier(solver='adam', alpha=1, hidden_layer_sizes=(5, 2),  random_state=1)

clfs = {'Logistic': logist, 'NN': clfnn}
#clfs = {'Logistic': logist, 'Maxent': 'maxent', 'SVC rbf': svmrbf, 'NeuralNet': clfnn2 }#, 'NN': clfnn}#, 'SVC': svmrbf}
#clfs = {'logistic': logist, 'SVC rbf': svmrbf}
#clfs = {'maxent': 'maxent', 'SVC' : svmrbf}

max_fact = 4000
frpsel = frp_data_subset()
features = frpsel.columns[4:,].tolist()
ffs = ['loss_last_sec', 'loss_this_prim', 'loss_accum_prim',
       'loss_accum_sec', 'loss_three_prim', 'loss_three_sec', 'f_prim',
       'gain', 'dem', 'dc_med', 'ffmc_med', 'fwi_med', 'ffmc_75p',
       'fwi_75p', 'dc_7mm', 'ffmc_7mm', 'fwi_7mm', 'dc_3m', 'latind']
feats =  [ features + ['lonind', 'latind'], ffs]
#features = ['dc_med', 'ffmc_med', 'fwi_med', 'ffmc_75p',
#       'fwi_75p', 'dc_7mm', 'ffmc_7mm', 'fwi_7mm', 'dc_3m']

#dfr = predict_years(clfs, frpsel, features + ['lonind', 'latind'], max_fact)
#dfrfwi = predict_years(clfs, frpsel, features, max_fact)

#feats = [['lonind', 'latind', 'loss_this', 'loss_last',
#         'loss_three', 'loss_accum', 'f_prim', 'gain', 'fwi', 'dc', 'ffmc'], ['dc', 'fwi', 'ffmc']]
#roc_plots(frpsel, feats, clfs, cv, 'test_v2_dem_2018_roll_sel_peat_depth', 8000)

#XS = frpsel[['lonind', 'latind', 'loss_last', 'loss_accum', 'loss_three', 'loss_this', 'f_prim', 'gain', 'fwi', 'dc', 'ffmc']]
#X_scaled = preprocessing.scale(XS.values)
