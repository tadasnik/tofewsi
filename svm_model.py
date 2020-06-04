from __future__ import print_function
import os
import geopandas as gpd
import calendar
import codecs, json
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1 import AxesGrid
from gridding import Gridder
import numpy as np
import pandas as pd
from scipy import interp
from random import randint
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc, accuracy_score, roc_auc_score, balanced_accuracy_score, average_precision_score, precision_recall_curve, brier_score_loss
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

def spatial_subset_dfr(dfr, bbox):
    dfr = dfr[(dfr['latitude'] < bbox[0]) &(dfr['latitude'] > bbox[2])]
    dfr = dfr[(dfr['longitude'] > bbox[1]) &(dfr['longitude'] < bbox[3])]
    return dfr


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
    #                         sharex = true, sharey = true,
    #                         subplot_kw={'projection': ccrs.platecarree()})
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
            #gl = colax.gridlines(ccrs.PlateCarree(), draw_labels=true)
            #gl.xlabels_top = False
            #colax.coastlines('50m')
            #colax.set_extent([9, 96, -10, 143], crs=ccrs.PlateCarree())
    tit = fig.suptitle('{0}'.format(year), y=.97, fontsize=18)
    plt.savefig('figs/models_probs_{}.png'.format(year), dpi=300)#, bbox_inches='tight', bbox_extra_artists=[tit])
    #plt.show()

class FireMod():
    def __init__(self, dfr, feature_columns, label_column, label_threshold, max_fact = None):
        self.max_fact = max_fact
        self.features = feature_columns
        self.label_column = label_column
        self.label_threshold = label_threshold
        self.forecast_path = '/mnt/data2/SEAS5/forecast'
        self.models = {
            'svmlin': svm.SVC(kernel='linear', probability=True, C=1,
                              random_state=random_state),
            'svmrbf': svm.SVC(kernel='rbf', C=1, gamma=0.15, probability=True,
                              random_state=random_state),
            'logist': LogisticRegression(solver = 'liblinear', penalty='l1', class_weight={0:1,1:5}),
            'clfnn1': MLPClassifier(solver='lbfgs', alpha=1,
                                    hidden_layer_sizes=(10),
                                    max_iter = 1000,
                                    activation='logistic', random_state=1),
            'clfnn2': MLPClassifier(solver='lbfgs', alpha=1,
                                    hidden_layer_sizes=(100),
                                    activation='logistic', random_state=1),
            'randf': RandomForestClassifier(n_estimators=100, max_depth=4,
                                 random_state=0),
            'clfnn': MLPClassifier(solver='lbfgs', alpha=2,
                                   hidden_layer_sizes=(14),
                                   activation='logistic', random_state=1)
        }
        self.cvms = {
            'stratified': StratifiedKFold(n_splits=10, shuffle = True),
        }
        self.rus = RandomUnderSampler(random_state=0)
        self.dfr = self.year_month(dfr)
        self.dfr['labels'] = self.class_labels(dfr)
        #self.dfr = self.dfr[self.dfr.peat_depth > 0]

    def year_month(self, dfr):
        dfr['year'] = dfr.date.dt.year
        dfr['month'] = dfr.date.dt.month
        #dfr['year'] = (((dfr.month.astype(int) - 1) // 12.) + 2002).astype(int)
        #dfr['month'] = (((dfr.month.astype(int) - 1) % 12.) + 1).astype(int)
        return dfr

    def class_labels(self, dfr):
        labels = np.zeros_like(dfr[self.label_column].values)
        labels[dfr[self.label_column] > self.label_threshold] = 1
        return labels

    def leave_year_split(self, dfr):
        year = 2002
        while year <= 2018:
            train, test = self.get_year_train_test(dfr, year)
            yield train, test
            year += 1

    def get_year_train_test(self, dfr, year):
        X_train_inds = np.where(dfr.year != year)[0]
        X_test_inds = np.where(dfr.year == year)[0]
        if self.max_fact:
            X_train_inds = np.random.choice(X_train_inds, size = self.max_fact)
        return X_train_inds, X_test_inds

    def equalize_classes(self, dfr):
        data_array, labels = self.rus.fit_resample(dfr, dfr['labels'])
        equalized_dfr = pd.DataFrame(data = data_array, columns = dfr.columns)
        return equalized_dfr

    def roc_plots(self, features, models, name):
        dfr = self.equalize_classes(self.dfr)
        feature_names = ['Fire weather', 'Land cover', 'Fire weather + land cover']
        model_names = ['SVM', 'NN']
        plot_x_size = len(features) * 4
        plot_y_size = len(models) * 4
        fig, axes = plt.subplots(nrows=len(features), ncols=len(models),
                                 figsize = (plot_y_size, plot_x_size),
                                 sharex = True, sharey = True)
        fig.subplots_adjust(left=0.05, right=0.97, bottom=0.05, hspace=0.2, wspace=0.1)
        for row_nr, rowax in enumerate(axes):
            for col_nr, colax in enumerate(rowax):
                #X = features[row_nr]
                #XX = feats_unscaled[row_nr]
                clf = self.models[models[col_nr]]
                print(clf)
                tprs, tprsint, fprs, aucs, mean_fpr, score  = self.do_roc_year(dfr, features[row_nr], clf, max_fact)
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
                if row_nr == len(features) - 1:
                    colax.set_xlabel('False Positive Rate')
                if col_nr == 0:
                    colax.set_ylabel('True Positive Rate')
                #if row_nr == 0:
                colax.set_title('{0} and {1} model'.format(feature_names[row_nr], model_names[col_nr]))
                colax.legend(loc="lower right")
        #fig.text(0.5, 0.01, 'Mean {}'.format(fwi_ds), ha='center', fontsize = 14)
        #fig.text(0.01, 0.5, 'Active fire pixel count', va='center', rotation='vertical', fontsize = 14)
        tit = fig.suptitle('ROC curves for different feature sets and models'.format(name), y=.97, fontsize=18)
        #plt.savefig('figs/rocs_{}_indonesia.png'.format(name), dpi=300)#, bbox_inches='tight', bbox_extra_artists=[tit])
        plt.savefig('figs/rocs_{0}_indonesia.png'.format(name), dpi=300)#, bbox_inches='tight', bbox_extra_artists=[tit])
        plt.show()

    def plot_features(self, features, dfr, months):
        year = dfr.year.values[0]
        titles = ['{0}/{1} - {2} month lead'.format(year, x, y +1) for y, x in enumerate(months)]
        ylabels = ['Climatology', 'SEAS5 model mean', 'Difference']
        #train = self.prepare_train(self.dfr)
        plot_y_size = len(features) * 5
        plot_x_size = (len(months) ) * 10
        projection = ccrs.PlateCarree()
        axes_class = (GeoAxes,
                      dict(map_projection = projection))
        fig = plt.figure(figsize=(plot_x_size, plot_y_size))
        axgr = AxesGrid(fig, 111, axes_class = axes_class,
                        nrows_ncols=(len(features), len(months)),
                        axes_pad=.3,
                        cbar_mode='edge',
                        cbar_location='right',
                        cbar_pad=0.5,
                        cbar_size='3%',
                        label_mode='')
        month_names = [calendar.month_abbr[x] for x in months]
        gri = Gridder(bbox = 'indonesia', step = 0.25)
        col_names = features
        for row_nr, rowax in enumerate(axgr.axes_row):
            for col_nr, colax in enumerate(rowax):
                dfr_sub = dfr[dfr.month == months[col_nr]]
                col_name = col_names[row_nr]
                vmin = 0.0
                vmax = 1.0
                ds = gri.dfr_to_dataset(dfr_sub, col_name, np.nan)
                pl=ds[col_name].plot.pcolormesh(ax=colax, transform=ccrs.PlateCarree(),
                                             x = 'longitude', y='latitude',
                                             add_colorbar=False, add_labels=False)
                colax.set_extent([94, 142, -11, 7], crs=ccrs.PlateCarree())
                if row_nr == 0:
                    colax.set_title(titles[col_nr])
                if col_nr == 0:
                    pass
                    #colax.set_ylabel(ylabels[row_nr])
        axgr.cbar_axes[0].colorbar(pl)
        axgr.cbar_axes[1].colorbar(pl)
        axgr.cbar_axes[2].colorbar(pl)
        #tit = fig.suptitle('{0}'.format(name), y=.97, fontsize=18)
        plt.show()


    def fit_model(self, train, features, model):
        train['labels'] = train['labels'].astype(int)
        trainf = train[features]
        scaler = preprocessing.StandardScaler().fit(trainf)
        train_scaled = scaler.transform(trainf.values)
        clf = self.models[model].fit(train_scaled, train['labels'])
        return clf, scaler

#

    def plot_forecast(self, features, clf, frp_clim, frp_s5, months, name):
        titles = ['2019/{0} - {1} month lead'.format(x, y +1) for y, x in enumerate(months)]
        ylabels = ['Climatology', 'SEAS5 model mean', 'Difference']
        #train = self.prepare_train(self.dfr)
        train, test = get_year_train_test_select(self.dfr, 2019)
        clf, scaler = self.fit_model(train, features, clf)
        clim_scaled = scaler.transform(frp_clim[features].values)
        s5_scaled = scaler.transform(frp_s5[features].values)
        frp_clim['probs'] = clf.predict_proba(clim_scaled)[:, 1]
        frp_s5['probs'] = clf.predict_proba(s5_scaled)[:, 1]
        frp_s5['probs_diff'] = frp_s5['probs'] - frp_clim['probs']

        dss = [frp_clim, frp_s5, frp_s5]

        plot_y_size = 3 * 5
        plot_x_size = (len(months) ) * 10
        projection = ccrs.PlateCarree()
        axes_class = (GeoAxes,
                      dict(map_projection = projection))
        fig = plt.figure(figsize=(plot_x_size, plot_y_size))
        axgr = AxesGrid(fig, 111, axes_class = axes_class,
                        nrows_ncols=(3, len(months)),
                        axes_pad=.3,
                        cbar_mode='edge',
                        cbar_location='right',
                        cbar_pad=0.5,
                        cbar_size='3%',
                        label_mode='')
        month_names = [calendar.month_abbr[x] for x in months]
        gri = Gridder(bbox = 'indonesia', step = 0.25)
        col_names = ['probs', 'probs', 'probs_diff']
        for row_nr, rowax in enumerate(axgr.axes_row):
            dfr = dss[row_nr]
            for col_nr, colax in enumerate(rowax):
                dfr_sub = dfr[dfr.month == months[col_nr]]
                col_name = col_names[row_nr]
                vmin = 0.0
                vmax = 1.0
                if row_nr == 2:
                    vmin = -1.0
                    vmax = 1.0
                ds = gri.dfr_to_dataset(dfr_sub, col_name, np.nan)
                pl=ds[col_name].plot.pcolormesh(ax=colax, transform=ccrs.PlateCarree(),
                                             vmin = vmin, vmax = vmax, x = 'longitude', y='latitude',
                                             add_colorbar=False, add_labels=False)
                colax.set_extent([94, 142, -11, 7], crs=ccrs.PlateCarree())
                if row_nr == 0:
                    colax.set_title(titles[col_nr])
                if col_nr == 0:
                    colax.set_ylabel(ylabels[row_nr])
                #gl = colax.gridlines(ccrs.PlateCarree(), draw_labels=true)
                #gl.xlabels_top = False
                #colax.coastlines('50m')
                #colax.set_extent([9, 96, -10, 143], crs=ccrs.PlateCarree())
        axgr.cbar_axes[0].colorbar(pl)
        axgr.cbar_axes[1].colorbar(pl)
        axgr.cbar_axes[2].colorbar(pl)
        #tit = fig.suptitle('{0}'.format(name), y=.97, fontsize=18)
        plt.savefig('figs/models_probs_{0}.png'.format(name), dpi=300, bbox_inches='tight')#, bbox_extra_artists=[tit])
        plt.show()


    def prepare_train(self, train):
        #train = self.equalize_classes(train)
        train = train.sample(n = self.max_fact)
        return train

    def do_roc_year(self, dfr, features, clf, max_fact):
        tprs = []
        tprsint = []
        fprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        for year in range(2002, 2019, 1):
            probas_, score, y_test = self.predict_year(dfr, features, year, clf)
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


    def predict_year(self, frpsel, features, year, clf):
        x_train, test = self.get_year_train_test(frpsel, year)
        #x_train = self.equalize_classes(frpsel.iloc[train, :])
        x_train = frpsel.iloc[x_train, :]
        y_train = x_train['labels']
        scaler = preprocessing.StandardScaler().fit(x_train[features])
        x_train_scaled = scaler.transform(x_train[features].values)
        x_test = frpsel.iloc[test, :]
        x_test_scaled = scaler.transform(x_test[features].values)
        y_test = x_test['labels']
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

    def predict(self, train, target, features, clf):
        train = self.equalize_classes(train)
        train = train.sample(n = self.max_fact)
        trainf = train[features]
        scaler = preprocessing.StandardScaler().fit(trainf)
        train_scaled = scaler.transform(trainf.values)
        target_scaled = scaler.transform(target[features].values)
        clf = self.models[clf]
        clf.fit(train_scaled, train['labels'])
        probas = clf.predict_proba(target_scaled)
        return probas

    def feature_selection(self, dfr, feats):
        dfrsel = self.equalize_classes(dfr)
        print(dfrsel.columns)
        custom_cv  = self.leave_year_split(dfrsel)
        XS = dfrsel[feats]
        X_scaled = preprocessing.scale(XS.values)
        labels = dfrsel['labels']
        rfe = RFE(estimator = self.models['logist'],
                  n_features_to_select=1, step=1)
        rfe.fit(X_scaled, dfrsel['labels'])
        print('feature ranking RFE: ', list(zip(feats, rfe.ranking_)))
        print("Optimal number of features : %d" % rfe.n_features_)
        n_estimators = 20
        clr = BaggingClassifier(self.models['logist'],
                                max_samples = 1.0 / n_estimators,
                                n_estimators=n_estimators, n_jobs=7)
        scores = cross_val_score(clr, X_scaled, labels, cv=custom_cv)
        print(scores)

        custom_cv  = self.leave_year_split(dfrsel)
        rfecv = RFECV(estimator = self.models['logist'], step=1, cv=custom_cv,
                      scoring = 'accuracy', n_jobs=7)
        #rfecv = RFECV(estimator=svmlin, step=1, cv=StratifiedKFold(9, shuffle = True),
        #              scoring='accuracy', n_jobs=7)
        rfecv.fit(X_scaled, dfrsel['labels'])
        print('feature ranking RFECV: ', list(zip(feats, rfecv.ranking_)))
        print("Optimal number of features : %d" % rfe.n_features_)
        print("Optimal number of features : %d" % rfecv.n_features_)
        print(rfecv.grid_scores_)
        return list(zip(feats, rfecv.ranking_))

    def equalize_classes_max(self, dfr):
        ones = dfr[dfr.labels == 1]
        if len(ones) > 2000:
            ones = ones.sample(n = 2000)
        zeros = dfr[dfr.labels == 0]
        if self.max_fact > len(ones):
            zeros_nr = self.max_fact - len(ones)
        else:
            print('to many ones, can not equalize')
            return None
        select = pd.concat([ones, zeros.sample(n = int(len(ones) * 1))])
        return select


    def leave_year_split_(self, dfr):
        year = 2002
        while year <= 2018:
            train, test = self.get_year_train_test_(dfr, year)
            yield train, test
            year += 1

    def get_year_train_test_(self, dfr, year):
        X_train = dfr[dfr.year != year]
        X_test = dfr[dfr.year == year]
        X_train = self.equalize_classes(X_train)
        #X_train = X_train.sample(n = self.max_fact)
        #if self.max_fact:
        #    X_train_inds = np.random.choice(X_train_inds, size = self.max_fact)
        return X_train.index.values, X_test.index.values


    def get_scores_(self, dfr, feats, model):
        #dfrsel = self.equalize_classes(dfr)
        #dfrsel = self.prepare_train(dfr)
        #dfrsel = dfrsel.sample(n = self.max_fact)
        dfrsel = dfr.copy()
        custom_cv  = self.leave_year_split_(dfrsel)
        cv = StratifiedKFold(n_splits=5, shuffle = True)
        XS = dfrsel[feats]
        labels = dfrsel['labels'].astype(int)
        X_scaled = preprocessing.scale(XS.values)
        scoring = {'AUC': 'roc_auc',
                   'bal_acc': metrics.make_scorer(metrics.balanced_accuracy_score),
                   'acc': metrics.make_scorer(metrics.accuracy_score),
                   'precision': metrics.make_scorer(metrics.precision_score),
                   'recall': metrics.make_scorer(metrics.recall_score),
                   'kappa': metrics.make_scorer(metrics.cohen_kappa_score)
                   }
        #scorer = metrics.make_scorer(metrics.recall_score)
        scores = cross_validate(self.models[model], X_scaled,
                                 labels, scoring = scoring, cv = cv)
        for key, item in scores.items():
            print(key, np.mean(item))

    def run_analysis(self, year, dfr, feats, model):
        train, test = self.get_year_train_test_(dfrsel, year)
        x_train = dfr.iloc[train, :][feats]
        x_test = dfr.iloc[test, :][feats]
        y_train = dfr.iloc[train, :]['labels'].astype(int)
        y_test = dfr.iloc[test, :]['labels'].astype(int)
        scaler = preprocessing.StandardScaler().fit(x_train)
        x_train_scaled = preprocessing.scale(x_train.values)
        x_test_scaled = preprocessing.scale(x_train.values)
        clf = self.models[model]
        clf.fit(x_train_scaled, y_train)
        probas = clf.predict_proba()



    def get_scores(self, dfr, feats, model):
        dfrsel = self.equalize_classes(dfr)
        custom_cv  = self.leave_year_split(dfrsel)
        XS = dfrsel[feats]
        labels = dfrsel['labels']
        X_scaled = preprocessing.scale(XS.values)
        scorer = metrics.make_scorer(metrics.precision_score)
        scores = cross_val_score(self.models[model],  X_scaled, labels, scoring = scorer, cv=custom_cv)
        print(scores)
        print(np.mean(scores))

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

def class_labels(frpsel, column, threshold):
    labels = np.zeros_like(frpsel[column].values)
    labels[frpsel[column] > threshold] = 1
    return labels

def feature_selection(frpsel, feats, max_fact):
    frpsel = balance_classes(frpsel, 'frp', 10)
    factor = subset_factor(frpsel.shape[0], max_fact)
    frpsel = frpsel.iloc[::factor, :]
    labels = class_labels(frpsel, 'frp', 10)
    custom_cv  = leave_year_split(frpsel)
    print('features', feats)
    XS = frpsel[feats]
    X_scaled = preprocessing.scale(XS.values)
    rfe = RFE(estimator = svmlin, n_features_to_select=13, step=1)
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

def get_year_train_test(frpsel, year, max_fact = None):
    years = (((frpsel.month.astype(int) - 1) // 12.) + 2002).astype(int)
    X_train_inds = np.where(years != year)[0]
    X_test_inds = np.where(years == year)[0]
    if max_fact:
        X_train_inds = np.random.choice(X_train_inds, size = max_fact)
    return X_train_inds, X_test_inds

def predict_probability(x_train, y_train, x_test, y_test, clf):
    clf.fit(x_train, y_train)
    probas = clf.predict_proba(x_test)
    return probas, clf.score(x_test, y_test)

def frp_data_subset(frp):
    gri = Gridder(bbox = 'indonesia', step = 0.25)
    frp = gri.spatial_subset_ind_dfr(frp, gri.bbox)
    return frp

def balance_classes(frp, column, threshold):
    #make sample sizes evenish
    frp1 = frp[frp[column] > threshold]
    frp0 = frp[frp[column] == 0]
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


def select_svm_params(frpsel, feats, max_fact):
    #features = ['fwi', 'dc', 'ffmc']
    #features = ['lonind', 'latind', 'loss_this', 'loss_last', 'loss_three', 'loss_accum', 'f_prim', 'gain', 'fwi', 'dc', 'ffmc']
    frpsel = balance_classes(frpsel, 'duration', 7)
    factor = subset_factor(frpsel.shape[0], max_fact)
    frpsel = frpsel.iloc[::factor, :]
    scaler = preprocessing.StandardScaler().fit(frpsel[feats])
    labels = class_labels(frpsel, 'duration', 7)
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
    return dfr

def get_year_train_test_select(dfr, year):
    X_train = dfr[dfr.year != year]
    X_test = dfr[dfr.year == year]
    poss = dfr[dfr.labels == 1]
    #negs = dfr[dfr.labels == 0].sample(n = poss.shape[0] * 10)
    negs = dfr[dfr.labels == 0]#.sample(n = poss.shape[0] * 10)
    X_train = pd.concat([poss, negs])
    return X_train, X_test

def forecast_validation(self, features, clf, month, name):
    frpfr = pd.read_parquet('/mnt/data/frp/monthly_frp_count_indonesia_{0}deg_2019_09.parquet'.format(self.res))
    frp_train = pd.read_parquet('data/feature_train_fr_0.25deg_v4.parquet')
    fm = FireMod(frp_train, features, 'frp', 10, max_fact  = 4000)
    train, test = get_year_train_test_select(fm.dfr, 2019)
    clf, scaler = fm.fit_model(train, 'clfnn1')
    year = 2019
    res = {}
    frp_s5 = pd.read_parquet(os.path.join(fm.forecast_path,
                                          '{0}_{1:02}/s5_features.parquet'.format(year, start_month)))
    frp_s5 = fm.year_month(frp_s5_new)
    frp_clim = pd.read_parquet(os.path.join(fm.forecast_path,
                                            '{0}_{1:02}/clim_features.parquet'.format(year, start_month)))
    frp_clim = fm.year_month(frp_clim)
    s5_scaled = scaler.transform(frp_s5[features].values)
    frp_s5_probs = clf.predict_proba(s5_scaled)[:, 1]
    clim_probs = clf.predict_proba(s5_scaled)[:, 1]
    res['init'] = '{0}-{1}'.format(year, start_month)

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

frp_train = pd.read_parquet('data/feature_train_fr_0.25deg_v4.parquet')
#frpsel = frp_data_subset(frp_train)
#frpsel = frp_train[frp_train.duration < 150]
#frpsel = frp_train[frp_train.year < 2019]
#frpsel = frpsel.reset_index()


#features = frpsel.columns[4:,].tolist()
misc = ['lonind', 'latind', 'month']

weather_this = ['tp_med', 't2m_med', 'w10_med', 'h2m_med',
           't2m_75p', 'w10_med', 'h2m_75p']

weather = ['tp_sum', 'tp_1', 'tp_2', 'tp_3', 'tp_4', 'tp_5', 't2m_med', 'h2m_med',
           'h2m_75p', 't2m_7mm', 'h2m_7mm',
           't2m_3sum', 'h2m_3sum']
weather_sub = ['tp_sum', 'tp_1', 'tp_2', 'tp_3', 'tp_4', 'tp_5',
                't2m_med', 't2m_1', 't2m_2', 't2m_3', 't2m_4', 't2m_5',
                'w10_med', 'w10_1', 'w10_2', 'w10_3', 'w10_4', 'w10_5',
                'h2m_med', 'h2m_1', 'h2m_2', 'h2m_3', 'h2m_4', 'h2m_5']


weather_past = ['tp_7mm', 't2m_7mm', 'w10_7mm', 'h2m_7mm',
           'tp_3sum', 't2m_3sum',  'h2m_3sum',
           'tp_3m']

fwi_this = ['dc_med', 'ffmc_med', 'fwi_med', 'dc_75p', 'ffmc_75p',
       'fwi_75p']

fwi_past = ['dc_7mm', 'ffmc_7mm', 'fwi_7mm',
       'dc_3sum', 'dc_6sum', 'ffmc_3sum', 'fwi_3sum']

fwi = ['dc_med', 'ffmc_med', 'fwi_med', 'dc_75p', 'ffmc_75p',
       'fwi_75p', 'dc_7mm', 'ffmc_7mm', 'fwi_7mm',
       'dc_3sum', 'ffmc_3sum', 'fwi_3sum']

lc = ['loss_last_prim', 'loss_last_sec', 'loss_prim_before', 'loss_sec_before',
       'loss_three_prim', 'loss_three_sec', 'frp_acc', 'gain', 'f_prim', 'dem', 'peatd']

#month = 7
#year = 2019
#features = weather+lc
#model = 'clfnn1'
#months = [7, 8, 9, 10, 11]

#frp_train = pd.read_parquet('data/feature_train_fr_0.25deg_v4.parquet')
#fm = FireMod(frp_train, features, 'frp', 10, max_fact  = 4000)
#frp = pd.read_parquet('/mnt/data2/SEAS5/forecast/frp_features_2019_10.parquet')
#train, test = get_year_train_test_select(fm.dfr, 2019)
#clf, scaler = fm.fit_model(train, features, 'clfnn1')
#res = {}
#frp_s5 = pd.read_parquet(os.path.join(fm.forecast_path,
#                                      '{0}_{1:02}/s5_features.parquet'.format(year, month)))
#frp_s5 = fm.year_month(frp_s5)
#frp_clim = pd.read_parquet(os.path.join(fm.forecast_path,
#                                        '{0}_{1:02}/clim_features.parquet'.format(year, month)))
#frp_clim = fm.year_month(frp_clim)
gri = Gridder(bbox = 'indonesia', step = 0.25)
"""
frp_s5 = pd.read_parquet('/mnt/data2/SEAS5/forecast/{0}_{1:02}/s5_features.parquet'.format(year, month))
frp_clim = pd.read_parquet('/mnt/data2/SEAS5/forecast/{0}_{1:02}/clim_features.parquet'.format(year, month))
frp_clim = fm.year_month(frp_clim)
frp_s5 = fm.year_month(frp_s5)
s5_scaled = scaler.transform(frp_s5[features].values)
clim_scaled = scaler.transform(frp_clim[features].values)
frp_s5.loc[:, 'probs'] = clf.predict_proba(s5_scaled)[:, 1]
frp_s5.loc[:, 'clim_probs'] = clf.predict_proba(clim_scaled)[:, 1]
"""

"""
for month in months:
    init = '{0}-{1}'.format(year, month)
    res[init] = {}
    res[init]['forecast'] = {}

for month in months:

    #clim
    frp_clim = pd.read_parquet('/mnt/data2/SEAS5/forecast/{0}_{1:02}/clim_features.parquet'.format(year, month))
    frp_clim = frp_clim.dropna()
    frp_clim = fm.year_month(frp_clim)
    clim_scaled = scaler.transform(frp_clim[features].values)
    frp_clim.loc[:, 'clim_probs'] = clf.predict_proba(clim_scaled)[:, 1]
    init = '{0}-{1}'.format(year, month)
    frp_clim['clim_probs'] = (frp_clim.clim_probs.values * 100).astype(int).tolist()
    res[init]['climatology'] = frp_clim[frp_clim.month == month]['clim_probs'].tolist()

    #forecast
    try:
        frp_s5 = pd.read_parquet('/mnt/data2/SEAS5/forecast/{0}_{1:02}/s5_features.parquet'.format(year, month))
    except:
        continue
    frp_s5 = fm.year_month(frp_s5)
    s5_scaled = scaler.transform(frp_s5[features].values)
    frp_s5.loc[:, 'probs'] = clf.predict_proba(s5_scaled)[:, 1]
    frps = frp[(frp.year == year) & (frp.month == month)][['lonind', 'latind', 'frp']]
    frps = pd.merge(frp_s5[frp_s5.month == month][['lonind', 'latind']], frps,
                                              on = ['lonind', 'latind'], how = 'left')
    frps = frps.fillna(0)
    if len(frps) > 0:
        res[init]['frp'] = frps['frp'].values.tolist()
    else:
        res[init]['frp'] = []
    for lead, month in enumerate(frp_s5['month'].unique()[:4], 1):
        #res[init]['forecast'][str(lead)] = {}
        probs = frp_s5[frp_s5.month == month]['probs']
        lead_res = (probs.values * 100).astype(int).tolist()
        init = '{0}-{1}'.format(year, month)
        try:
            res[init]['forecast'][str(lead)] = lead_res
        except KeyError:
            res[init] = {}
            res[init]['forecast'] = {}
            res[init]['forecast'][str(lead)] = lead_res
        #if (month == 7) & (lead == 1):
        #    frps['latitude'] = gri.lat_bins[frps.latind.values + 1]
        #    frps['longitude'] = gri.lon_bins[frps.lonind.values]
        #    frps[['longitude', 'latitude']].to_json('/home/tadas/tofewsi/website/assets/geo/lonlats_fore.json', orient="values")

with open('/home/tadas/tofewsi/website/assets/forecast_.json', 'w') as outfile:
    json.dump(res, outfile)


#fm.plot_forecast(features, 'clfnn1', frp_clim, frp_s5,  [9, 10, 11], 'clfnn_test')
"""
