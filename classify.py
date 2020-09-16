import numpy as np
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ranksums
import json
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from sklearn.metrics import precision_recall_curve, roc_curve, auc, fbeta_score
from imblearn.metrics import geometric_mean_score
from ArgsClassify import *  # Parameters for training classifier


def evaluate(X, y, estm):
    # Performance metrics
    y_pred = estm.predict(X)
    print(confusion_matrix(y, y_pred).ravel())
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    # ROC curve
    try:
        if "decision_function" not in dir(estm):
            y_prob = estm.predict_proba(X)[:, 1]
        else:
            y_prob = estm.decision_function(X)
        pre, rec, _ = precision_recall_curve(y, y_prob)
        fpr, tpr, _ = roc_curve(y, y_prob)
        aucroc = auc(fpr, tpr)
        aucpr = auc(rec, pre)
    except AttributeError:
        print("Classifier don't have predict_proba or decision_function, ignoring roc_curve.")
        pre, rec = None, None
        fpr, tpr = None, None
        aucroc = None
        aucpr = None
    eval_dictionary = {
        "CM": confusion_matrix(y, y_pred),  # Confusion matrix
        "ACC": (tp + tn) / (tp + fp + fn + tn),  # accuracy
        "F1": fbeta_score(y, y_pred, beta=1),
        "F2": fbeta_score(y, y_pred, beta=2),
        "GMean": geometric_mean_score(y, y_pred, average='binary'),
        "SEN": tp / (tp + fn),
        "PREC": tp / (tp + fp),
        "SPEC": tn / (tn + fp),
        "MCC": matthews_corrcoef(y, y_pred),
        "PRCURVE": {"precision": pre, "recall": rec, "aucpr": aucpr},
        "ROCCURVE": {"fpr": fpr, "tpr": tpr, "aucroc": aucroc}
    }
    return eval_dictionary


def train_gridCV_imb(X, y, estm, sampler, param_grid, scoring="recall"):
    if sampler is not None:
        X_res, y_res = sampler.fit_resample(X, y)
    else:
        X_res, y_res = X, y
    grid = GridSearchCV(estm, param_grid,
                        scoring=scoring, cv=5, n_jobs=-1)  # use recall for finding results
    grid.fit(X_res, y_res)
    return grid.best_estimator_, grid.best_params_, grid.best_score_, X_res, y_res


def imb_classification(X_tr, y_tr, X_te, y_te, imb_samplers, estimator,
                       grid_params=None, X_u=None, iteration=16, scoring="recall"):
    # Load data
    print("==============================================================")
    print(
        "Number of training samples: %d, Positive: %d, Negative: %d" % (len(y_tr), y_tr.sum(), len(y_tr) - y_tr.sum()))
    print("Number of test samples: %d, Positive: %d, Negative: %d" % (len(y_te), y_te.sum(), len(y_te) - y_te.sum()))

    # Imbalanced learning with different imbalance strategies
    perf_df = []
    pr_curves_all, roc_curves_all = {}, {}
    best_estimators_dict, best_params_dict = {}, {}
    for sampler_name, sampler in imb_samplers.items():
        # Train with sampled data
        acc_all, sen_all, prec_all, spec_all = [], [], [], []
        f1_all, f2_all, gmean_all = [], [], []
        mcc_all, aucpr_all, aucroc_all = [], [], []
        pr_cur, roc_cur = None, None  # Not support for mean PR-curve yet
        best_estimator, best_params = None, None
        # determine the iteration of sampler using randomized strategy
        if 'random_state' not in dir(sampler):
            iter_use = 1
        else:
            if sampler.random_state is not None:
                print("Perform single_iter sampling since random_state is set.")
                iter_use = 1
            else:
                iter_use = iteration
        print("Module of sampling strategy: {}, Evaluating with {:2d} iterations".format(sampler_name, iter_use))
        for ii in range(iter_use):
            if 'random_state' in dir(sampler):
                if sampler.random_state is None:
                    sampler.random_state = 0 + ii
            if X_u is None:
                estm, params, _, _, _ = train_gridCV_imb(X_tr, y_tr, estimator, sampler, grid_params, scoring=scoring)
            # For the TriTraining only
            else:
                if sampler is not None:
                    X_r, y_r = sampler.fit_resample(X_tr, y_tr)
                else:
                    X_r, y_r = X_tr, y_tr
                estimator.fit(X_r, y_r, X_u)
                estm = estimator
                params = None
            eval_d = evaluate(X_te, y_te, estm)  # evaluate with test data
            # judge the best estimator
            if len(acc_all) == 0:
                best_estimator = estm
                best_params = params
            else:
                best_estimator = estm if eval_d['SEN'] > sen_all[-1] else best_estimator
                best_params = params if eval_d['SEN'] > sen_all[-1] else best_params
            acc_all.append(eval_d['ACC'])
            f1_all.append(eval_d['F1'])
            f2_all.append(eval_d['F2'])
            gmean_all.append(eval_d['GMean'])
            sen_all.append(eval_d['SEN'])
            prec_all.append(eval_d['PREC'])
            spec_all.append(eval_d['SPEC'])
            mcc_all.append(eval_d['MCC'])
            pr_cur = eval_d['PRCURVE']
            roc_cur = eval_d['ROCCURVE']
            aucpr_all.append(pr_cur['aucpr'])
            aucroc_all.append(roc_cur['aucroc'])

        perf_df.append({
            "Sampler": sampler_name,
            "ACC(%)": "{:.2f}".format(np.mean(acc_all) * 100),
            "F1(%)": "{:.2f}".format(np.mean(f1_all) * 100),
            "F2(%)": "{:.2f}".format(np.mean(f2_all) * 100),
            "GMean(%)": "{:.2f}".format(np.mean(gmean_all) * 100),
            "SEN(%)": "{:.2f}".format(np.mean(sen_all) * 100),
            "PREC(%)": "{:.2f}".format(np.mean(prec_all) * 100),
            "SPEC(%)": "{:.2f}".format(np.mean(spec_all) * 100),
            "MCC(%)": "{:.2f}".format(np.mean(mcc_all) * 100),
            "AUCPR(%)": ("{:.2f}".format(np.mean(aucpr_all) * 100) if X_u is None else "Unavailable"),
            "AUCROC(%)": ("{:.2f}".format(np.mean(aucroc_all) * 100) if X_u is None else "Unavailable")
        })

        pr_curves_all[sampler_name] = pr_cur
        roc_curves_all[sampler_name] = roc_cur
        best_estimators_dict[sampler_name] = best_estimator
        best_params_dict[sampler_name] = best_params
    perf_df = pd.DataFrame(perf_df)
    print("Table: Performance of selected samplers")
    print(perf_df)
    return pr_curves_all, roc_curves_all, perf_df, best_estimators_dict, best_params_dict


"""
Function for feature selection with wilcoxon ranksum
Input:
    df_1: input dataframe 1
    df_2: input dataframe 2
    features_ind: features for selection
Output:
    ind_selected: selected features
"""


def feature_selection_ranksum(df_1, df_2, features_ind):
    # Column index should be the same at df_1 and df_2
    assert False not in (df_1.columns == df_2.columns)
    # Leave only the features of peptides here
    ind_selected = []
    pval_features = []
    for fi in features_ind:
        pval = ranksums(df_1[fi], df_2[fi]).pvalue
        pval_features.append(pval)
        if pval < .05:
            ind_selected.append(fi)
    pval_all = pd.DataFrame({"Feature": features_ind, "pvalue": pval_features})
    return ind_selected, pval_all


"""
Function for plot pr/roc curves
"""


def plot_pr_roc(pr_curves_all, roc_curves_all):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for sampler_name, curve_data in pr_curves_all.items():
        prec = curve_data['precision']
        recall = curve_data['recall']
        aucpr = curve_data['aucpr']
        ax[0].plot(recall, prec, lw=1.2, label="{:s} (AUC = {:.3f})".format(sampler_name, aucpr))
    ax[0].legend()
    ax[0].set_title("Precision-Recall Curve")
    ax[0].set_xlabel("recall")
    ax[0].set_ylabel("precision")
    ax[0].set_xlim([-0.05, 1.05])
    ax[0].set_ylim([-0.05, 1.05])
    for sampler_name, curve_data in roc_curves_all.items():
        fpr = curve_data['fpr']
        tpr = curve_data['tpr']
        aucroc = curve_data['aucroc']
        ax[1].plot(fpr, tpr, lw=1.2, label="{:s} (AUC = {:.3f})".format(sampler_name, aucroc))
    ax[1].legend()
    ax[1].set_title("Receiver Operating Curve")
    ax[1].set_xlabel("fpr")
    ax[1].set_ylabel("tpr")
    ax[1].set_xlim([-0.05, 1.05])
    ax[1].set_ylim([-0.05, 1.05])
    return fig


if __name__ == "__main__":
    if not os.path.exists("results_classification"):
        os.mkdir("results_classification")
    time_now = int(round(time.time() * 1000))
    time_now = time.strftime("%Y-%m-%d_%H-%M", time.localtime(time_now / 1000))
    cls_dir = "results_classification/{}".format(time_now)
    os.makedirs(cls_dir)
    with open(os.path.join(cls_dir, "arguments.txt"), 'w') as file:
        file.write("SELECT_FEATURES: {:s}.\n".format("Yes" if SELECT_FEATURES else "No"))
        file.write("Used models:\n")
        file.write("{:s}\n".format(str(model_dicts)))
        file.write("imbalance strategies:\n")
        file.write("{:s}\n".format(str(imb_strategies)))
    # Make dataset
    training_sets = {
        lab: pd.read_csv("data/train_data/{:s}_train.csv".format(lab))
        for lab in ["Anti-CoV", "Anti-Virus", "non-AVP", "non-AMP"]
    }
    test_sets = {
        lab: pd.read_csv("data/test_data/{:s}_test.csv".format(lab))
        for lab in ["Anti-CoV", "Anti-Virus", "non-AVP", "non-AMP"]
    }

    training_sets['All-AMP'] = pd.concat([training_sets['Anti-Virus'],
                                          training_sets['non-AVP']], axis=0)
    test_sets['All-AMP'] = pd.concat([test_sets['Anti-Virus'],
                                      test_sets['non-AVP']], axis=0)
    training_sets['All-Neg'] = pd.concat([training_sets['Anti-Virus'],
                                          training_sets['non-AVP'],
                                          training_sets['non-AMP']], axis=0)
    test_sets['All-Neg'] = pd.concat([test_sets['Anti-Virus'],
                                      test_sets['non-AVP'],
                                      test_sets['non-AMP']], axis=0)
    # Anti-CoV versus the other
    for nlab in cls_categories:
        nlab_dir = os.path.join(cls_dir, nlab)
        os.mkdir(nlab_dir)
        # Insert labels
        if nlab == "Pre-Cls":
            training_sets["Anti-CoV"].loc[:, 'Label'] = 1
            training_sets["Anti-Virus"].loc[:, 'Label'] = 1
            training_sets["non-AVP"].loc[:, 'Label'] = 1
            training_sets["non-AMP"].loc[:, 'Label'] = 0
            test_sets["Anti-CoV"].loc[:, 'Label'] = 1
            test_sets["Anti-Virus"].loc[:, 'Label'] = 1
            test_sets["non-AVP"].loc[:, 'Label'] = 1
            test_sets["non-AMP"].loc[:, 'Label'] = 0
            total_train = pd.concat([training_sets[plab] for plab in ['Anti-CoV', 'Anti-Virus', 'non-AVP', 'non-AMP']])
            total_test = pd.concat([test_sets[plab] for plab in ['Anti-CoV', 'Anti-Virus', 'non-AVP', 'non-AMP']])
        else:
            print("Make Classification for Anti-CoV versus {:s}".format(nlab))
            training_sets['Anti-CoV'].loc[:, 'Label'] = 1
            training_sets[nlab].loc[:, 'Label'] = 0
            test_sets['Anti-CoV'].loc[:, 'Label'] = 1
            test_sets[nlab].loc[:, 'Label'] = 0
            # Create total set
            total_train = pd.concat([training_sets[nlab], training_sets['Anti-CoV']])
            total_test = pd.concat([test_sets[nlab], test_sets['Anti-CoV']])
        features_inds = training_sets['Anti-CoV'].columns[2:-1]
        features_categories = ['AAC', 'DiC', 'gap', 'Xc', 'PHYC']
        cnt_all = {fc: len(list(filter(lambda x: True if fc in x else False, features_inds)))
                   for fc in features_categories[:-1]}
        cnt_all[features_categories[-1]] = len(features_inds) - sum(cnt_all.values())
        if SELECT_FEATURES:
            # Feature selection
            if nlab == 'Pre-Cls':
                df_pos = pd.concat([training_sets[plab] for plab in ['Anti-CoV', 'Anti-Virus', 'non-AVP']])
                df_neg = training_sets['non-AMP']
                selected_inds, df_pval = feature_selection_ranksum(df_pos, df_neg, features_ind=features_inds)
            else:
                selected_inds, df_pval = feature_selection_ranksum(training_sets['Anti-CoV'], training_sets[nlab],
                                                                   features_ind=features_inds)
            cnt_slt = {fc: len(list(filter(lambda x: True if fc in x else False, selected_inds)))
                       for fc in features_categories[:-1]}
            cnt_slt[features_categories[-1]] = len(selected_inds) - sum(cnt_slt.values())
            # Save the printed features
            smry_features = "Number of selected AAC features: {:d}/{:d}, DiC features: {:d}/{:d}, \n" \
                            "CKSAAGP features: {:d}/{:d}, PAAC features: {:d}/{:d}, PHYC features: {:d}/{:d}". \
                format(cnt_slt['AAC'], cnt_all['AAC'],
                       cnt_slt['DiC'], cnt_all['DiC'],
                       cnt_slt['gap'], cnt_all['gap'],
                       cnt_slt['Xc'], cnt_all['Xc'],
                       cnt_slt['PHYC'], cnt_all['PHYC'])
            print(smry_features)
            with open(os.path.join(nlab_dir, "feature_report.txt"), 'w') as file:
                file.write(smry_features)
            df_pval.to_csv(os.path.join(nlab_dir, "feature_pval.csv"), index=False)
            features_inds = selected_inds
        else:
            smry_features = "Number of AAC features: {:d}, DiC features: {:d}, \n" \
                            "CKSAAGP features: {:d}, PAAC features: {:d}, PHYC features: {:d}". \
                format(cnt_all['AAC'], cnt_all['DiC'], cnt_all['gap'], cnt_all['Xc'], cnt_all['PHYC'])
            print(smry_features)
            with open(os.path.join(nlab_dir, "feature_report.txt"), 'w') as file:
                file.write(smry_features)
        # Create Data for Training/Test
        X_train = total_train[features_inds].to_numpy()
        X_test = total_test[features_inds].to_numpy()
        y_train = total_train['Label'].to_numpy()
        y_test = total_test['Label'].to_numpy()
        # Load data
        with open(os.path.join(nlab_dir, "sample_report.txt"), 'w') as file:
            file.write("Number of training samples: %d, Positive: %d, Negative: %d \n" %
                       (len(y_train), y_train.sum(), len(y_train) - y_train.sum()))
            file.write("Number of test samples: %d, Positive: %d, Negative: %d \n" %
                       (len(y_test), y_test.sum(), len(y_test) - y_test.sum()))
        if NORMALISE:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
        # Train classifiers with imbalanced samplers
        for mdl_name, mdl in model_dicts.items():
            mdl_dir = os.path.join(nlab_dir, mdl_name)
            os.mkdir(mdl_dir)
            clf = mdl['model']  # For sklearn, clf.fit() will override the previous parameters
            clf_params = mdl['param_grid']
            if 'imblearn.ensemble' not in clf.__module__:
                prcs, rocs, perfs, estms, params_estm = imb_classification(X_train, y_train, X_test, y_test,
                                                                           imb_samplers=imb_strategies, estimator=clf,
                                                                           grid_params=clf_params, iteration=5)
            else:
                prcs, rocs, perfs, estms, params_estm = imb_classification(X_train, y_train, X_test, y_test,
                                                                           imb_samplers={"Default": None},
                                                                           estimator=clf,
                                                                           grid_params=clf_params, iteration=5)

            # Save the estimators, parameters, roc/pr results
            pr_roc_fig = plot_pr_roc(prcs, rocs)
            pr_roc_fig.savefig(os.path.join(mdl_dir, "pr_roc.png"), dpi=340)
            plt.clf()
            perfs.to_csv(os.path.join(mdl_dir, "performances.csv"), index=False)
            for spl_name in prcs.keys():
                spl_dir = os.path.join(mdl_dir, spl_name)
                os.mkdir(spl_dir)
                np.save(os.path.join(spl_dir, "pr.npy"), prcs[spl_name])
                np.save(os.path.join(spl_dir, "roc.npy"), rocs[spl_name])
                joblib.dump(estms[spl_name], os.path.join(spl_dir, "estimator.joblib"))
                json.dump(params_estm[spl_name], open(os.path.join(spl_dir, "params.json"), 'w'))
                # TODO: For combined negatives, save the results for different subsets
                if nlab == 'All-Neg' or nlab == 'All-AMP':
                    for d_index in ['non-AVP', 'Anti-Virus']:
                        X_test_sub = pd.concat([(test_sets[d_index])[features_inds],
                                                (test_sets['Anti-CoV'])[features_inds]]).to_numpy()
                        y_test_sub = np.concatenate([np.zeros(test_sets[d_index].shape[0]),
                                                     np.ones(test_sets['Anti-CoV'].shape[0])])
                        eval_dict_sub = evaluate(X_test_sub, y_test_sub, estms[spl_name])
                        with open(os.path.join(spl_dir, "subeval_{:s}.txt".format(d_index)), 'w') as fre:
                            fre.write("Results for {:s}".format(d_index))
                            fre.write(str(eval_dict_sub))
                if nlab == 'All-Neg':
                    X_test_sub = pd.concat([(test_sets['non-AMP'])[features_inds],
                                            (test_sets['Anti-CoV'])[features_inds]]).to_numpy()
                    y_test_sub = np.concatenate([np.zeros(test_sets['non-AMP'].shape[0]),
                                                 np.ones(test_sets['Anti-CoV'].shape[0])])
                    eval_dict_sub = evaluate(X_test_sub, y_test_sub, estms[spl_name])
                    with open(os.path.join(spl_dir, "subeval_non-AMP.txt"), 'a+') as fre:
                        fre.write("Results for non-AMP")
                        fre.write(str(eval_dict_sub))
                if nlab == 'Pre-Cls':
                    for d_index in ['non-AVP', 'Anti-Virus', 'Anti-CoV']:
                        X_test_sub = pd.concat([(test_sets[d_index])[features_inds],
                                                (test_sets['non-AMP'][features_inds])]).to_numpy()
                        y_test_sub = np.concatenate([np.ones(test_sets[d_index].shape[0]),
                                                     np.zeros(test_sets['non-AMP'].shape[0])])
                        eval_dict_sub = evaluate(X_test_sub, y_test_sub, estms[spl_name])
                        with open(os.path.join(spl_dir, "subeval_{:s}.txt".format(d_index)), 'a+') as fre:
                            fre.write("Results for non-AMP")
                            fre.write(str(eval_dict_sub))
