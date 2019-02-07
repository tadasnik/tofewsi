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
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score

def roc_plots(features, clfs, y, cv, name):
    plot_y_size = len(features) * 4
    plot_x_size = len(clfs) * 4
    fig, axes = plt.subplots(nrows=len(features), ncols=len(clfs),
                             figsize = (plot_y_size, plot_x_size),
                             sharex = True, sharey = True)
    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.05, hspace=0.2, wspace=0.1)
    mod_names = list(clfs.keys())
    for row_nr, rowax in enumerate(axes):
        for col_nr, colax in enumerate(rowax):
            print(row_nr, col_nr)
            X = features[row_nr]
            clf = clfs[mod_names[col_nr]]
            tprs, tprsint, fprs, aucs, mean_fpr  = do_roc(X, y, clf)
            for item in zip(fprs, tprs):
               colax.plot(item[0], item[1], lw=1, alpha=0.3)#,
            #         label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

            #i += 1
            colax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='0.6', alpha = .8)
            #         label='Chance', alpha=.8)
            mean_tpr = np.mean(tprsint, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            colax.plot(mean_fpr, mean_tpr, color='b',
                     label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                     lw=2, alpha=.8)

            std_tpr = np.std(tprsint, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            colax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                             label=r'$\pm$ 1 std. dev.')

            colax.set_xlim([-0.05, 1.05])
            colax.set_ylim([-0.05, 1.05])
            if row_nr == 2:
                colax.set_xlabel('False Positive Rate')
            if col_nr == 0:
                colax.set_ylabel('True Positive Rate')
            if row_nr == 0:
                colax.set_title(mod_names[col_nr])
            colax.legend(loc="lower right")
    #fig.text(0.5, 0.01, 'Mean {}'.format(fwi_ds), ha='center', fontsize = 14)
    #fig.text(0.01, 0.5, 'Active fire pixel count', va='center', rotation='vertical', fontsize = 14)
    tit = fig.suptitle('{0}'.format(name), y=.97, fontsize=18)
    plt.savefig('figs/rocs_{}.png'.format(name), dpi=300)#, bbox_inches='tight', bbox_extra_artists=[tit])

def do_roc(X, y, clf):
    tprs = []
    tprsint = []
    fprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for train, test in cv.split(X, y):
        probas_ = clf.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        fprs.append(fpr)
        tprs.append(tpr)
        tprsint.append(interp(mean_fpr, fpr, tpr))
        tprsint[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
    return tprs, tprsint, fprs, aucs, mean_fpr


#frp = pd.read_parquet('data/feature_frame.parquet')
#frp = pd.read_parquet('data/feature_frame_sumatra.parquet')
#frp = frp[(frp.lonind > 43) & (frp.lonind < 275)]
#frp = frp[(frp.lonind > 100) & (frp.lonind < 255)]
#frp = frp[(frp.latind > 141) & (frp.latind < 380)]
#frp = frp[(frp.latind > 200) & (frp.latind < 300)]
#sumatra = [6, 96, -6, 106]
java = [-5.8, 105, -9.3, 119]
sumatra = [6, 96, -6, 106]
borneo = [3, 108.7, -4.2, 119]
papa = [0, 130, -9.26, 141.5]
sulewasi = [1.5, 118.8, -5.6, 125.3]
riau = [3, 99, -2, 104]
bboxes = {'Java_Bali': java,
          'Sumatra': sumatra,
          'Borneo': borneo,
          'Papa': papa,
          'Sulewasi': sulewasi,
          'Riau': riau}
random_state = np.random.RandomState(0)
cv = StratifiedKFold(n_splits=10, shuffle = True)
svmlin = svm.SVC(kernel='linear', probability=True,
                     random_state=random_state)
svmrbf = svm.SVC(kernel='rbf', gamma='auto', probability=True,
                     random_state=random_state)
logist = LogisticRegression(solver = 'lbfgs')

clfs = {'logistic': logist, 'SVC linear': svmlin,'SVC rbf': svmrbf}


for key, item in bboxes.items():
    print(key)
    frp = pd.read_parquet('data/feature_frame_0.25deg.parquet')
    gri = Gridder(bbox = 'indonesia', step = 0.25)
    frp = gri.spatial_subset_ind_dfr(frp, item)

#make sample sizes evenish
    frp1 = frp[frp.frp > 10]
    frp0 = frp[frp.frp == 0]
    factor = int(frp0.shape[0] / frp1.shape[0])
    frpsel = pd.concat([frp1, frp0.iloc[::factor, :]])
    print(frpsel.shape[0])
    factor = int(frpsel.shape[0] // 4000)
    print(factor)
    if factor == 0:
        factor = 1
    frpsel = frpsel.iloc[::factor, :].copy()
    print(frpsel.shape[0])
#year = ((frpsel.month.astype(int) // 12.) + 2002).astype(int)
#groups = np.zeros_like(year)
#groups[year == 2015] = 1


    labels = frpsel['frp'].values.copy()
    labels[labels > 0] = 1

    feats = [frpsel[['fwi', 'dc', 'ffmc']],
             frpsel[['loss_this', 'loss_last', 'loss_three', 'loss_accum', 'f_prim', 'gain', 'fwi', 'dc', 'ffmc']],
             frpsel[['lonind', 'latind', 'loss_this', 'loss_last', 'loss_three', 'loss_accum', 'f_prim', 'gain', 'fwi', 'dc', 'ffmc']]]
    features = [preprocessing.scale(x.values) for x in feats]
    y = labels.copy()

    roc_plots(features, clfs, y, cv, key)

    XS = frpsel[['lonind', 'latind', 'loss_last', 'loss_accum', 'loss_three', 'loss_this', 'f_prim', 'gain', 'fwi', 'dc', 'ffmc']]
    X_scaled = preprocessing.scale(XS.values)
    rfe = RFE(estimator=svmlin, n_features_to_select=1, step=1)
    rfe.fit(X_scaled, y)
    print(key + ' feature ranking RFE: ', rfe.ranking_)

    n_estimators = 20
    clr = BaggingClassifier(svmlin,
                                 max_samples = 1.0 / n_estimators, n_estimators=n_estimators, n_jobs=7)
    scores = cross_val_score(clr, X_scaled, y, cv=9)
    print(scores)

    rfecv = RFECV(estimator=svmlin, step=1, cv=StratifiedKFold(9, shuffle = True),
                  scoring='accuracy', n_jobs=7)
    rfecv.fit(X_scaled, y)
    print(key + ' feature ranking RFECV: ', rfe.ranking_)
    print("Optimal number of features : %d" % rfecv.n_features_)
    print(rfecv.grid_scores_)




#XS = frpsel[['lonind', 'latind', 'loss_last', 'loss_accum', 'loss_three', 'loss_this', 'f_prim', 'gain', 'fwi', 'dc', 'ffmc']]
#X_scaled = preprocessing.scale(XS.values)

"""
#clf = svm.SVC(gamma = 'auto')#, class_weight = {1: 27})
clf = svm.SVC(kernel = 'linear')#, C=1)#, class_weight = {1: 27})
clf.fit(X_scaled[groups==0], labels[groups==0])
pred = clf.predict(X_scaled[groups==1])
print(accuracy_score(labels[groups==1], pred))

year = ((frpsel.month.astype(int) // 12.) + 2002).astype(int)
groups = np.zeros_like(year)
groups[year == 2015] = 1

n_estimators = 20
clr = BaggingClassifier(svm.SVC(gamma = 'auto', probability=False),
                             max_samples = 1.0 / n_estimators, n_estimators=n_estimators, n_jobs=7)
scores = cross_val_score(clr, X_scaled, labels, cv=9)

rfe = RFE(estimator=clf, n_features_to_select=1, step=1)
rfe.fit(X_scaled, labels)
print(rfe.ranking_)

rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(9, shuffle = True),
              scoring='accuracy', n_jobs=7)
rfecv.fit(X_scaled, labels)
print("Optimal number of features : %d" % rfecv.n_features_)
rfecv.grid_scores_

labels = frpsel['frp'].values.copy()
labels[labels > 0] = 1
#feat = frpsel[['lonind', 'latind', 'loss_last', 'loss_accum', 'loss_three',
#               'loss_this', 'f_prim', 'gain', 'fwi', 'dc', 'ffmc']]
feat = frpsel[[ 'loss_three', 'f_prim', 'fwi', 'dc', 'ffmc']]
#feat = frpsel[['fwi', 'dc', 'ffmc']]
X_scaled = preprocessing.scale(feat.values)

clf = svm.SVC(gamma = 'auto')#, class_weight = {1: 27})
#clf = svm.SVC(kernel = 'linear')#, C=1)#, class_weight = {1: 27})
clf.fit(X_scaled[groups==0], labels[groups==0])
pred = clf.predict(X_scaled[groups==1])
print(accuracy_score(labels[groups==1], pred))


#clf = svm.SVC(kernel = 'linear')#, C=1)#, class_weight = {1: 27})
#clf = linear_model.SGDClassifier(loss="hinge", penalty="l2", max_iter = 100)
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
sss.get_n_splits(X_scaled, labels)

clf = svm.SVC(kernel = 'linear')#, C=1)#, class_weight = {1: 27})
rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(5),
              scoring='accuracy')
rfecv.fit(X_scaled, labels)

scores = cross_val_score(clr, X_scaled, labels, cv=5)

print(rfecv.ranking_)
print("Optimal number of features : %d" % rfecv.n_features_)

for train_index, test_index in list(sss.split(X_scaled, labels)):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    Y_train, Y_test = labels[train_index], labels[test_index]
    #rfecv = RFECV(estimator=clf, step=1, cv=sss,
    #              scoring='accuracy')
    #rfecv.fit(X_train, Y_train)

    #print("Optimal number of features : %d" % rfecv.n_features_)
    selector = RFE(clf, 5)#, step = 1)
    selector.fit(X_train, Y_train)
    pred = selector.predict(X_test)
    print(accuracy_score(Y_test, pred))
    print(selector.support_)
    print(selector.ranking_)

#n_estimators = 100
#clr = BaggingClassifier(svm.SVC(kernel='linear', probability=False, class_weight='auto'),
#                             max_samples = 1.0 / n_estimators, n_estimators=n_estimators)

#clf = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', probability=True, class_weight='auto'), 
#                                            max_samples = 1.0 / n_estimators, n_estimators=n_estimators))
#clf = svm.SVC(gamma='scale', class_weight = {1: 27})
#clf.fit(X, y)
#clf.fit(X_scaled, labels)
"""



