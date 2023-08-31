from tqdm import tqdm

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def split_and_score(model, X, y, idx_splits, return_train=False):
    train_perfs = []
    val_perfs = []
    
    for train_idx, test_idx in idx_splits:
        model.fit(X[train_idx,:], y[train_idx])
        
        if return_train:
            train_perf = model.score(X[train_idx,:], y[train_idx])
            train_perfs.append(train_perf)

        val_perf = model.score(X[test_idx,:], y[test_idx])
        val_perfs.append(val_perf)
        
    if return_train:
        return np.array(train_perfs).T.reshape(1,-1), np.array(val_perfs).T.reshape(1,-1)
    else:
        return np.array(val_perfs).T.reshape(1,-1)

def dropcol_iterate(model, X, y, n_splits=3, col_order=None, scoring="accuracy", random_state=42, n_jobs=-1):
    X_orig = X.copy()
    X = X.copy()

    idx_splits = [s for s in KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(X)]
    if scoring == "sketchy":
        perf_init = split_and_score(model, X, y, idx_splits)
    else:
        perf_init = cross_val_score(model, X, y, scoring=scoring, cv=idx_splits, n_jobs=n_jobs).reshape(1,-1)
    perf_mean = np.mean(perf_init)
    
    perfs = []
    col_idxs = []
    drop_ind = []
    
    if col_order is not None:
        iterator = tqdm(col_order)
    else:
        iterator = tqdm(range(X.shape[1]))
    for col_idx in iterator:
        col_tmp = X[:,col_idx].copy()
        X[:,col_idx] = 0
        
        if scoring == "sketchy":
            perf_new = split_and_score(model, X, y, idx_splits)
        else:
            perf_new = cross_val_score(model, X, y, scoring=scoring, cv=idx_splits, n_jobs=n_jobs).reshape(1,-1)
        
        col_idxs.append(col_idx)
        perfs.append(perf_new)
        
        perf_mean_new = np.mean(perf_new)
        
        if perf_mean > perf_mean_new:
            X[:,col_idx] = col_tmp
            drop_ind.append(False)
        else:
            perf_mean = perf_mean_new
            drop_ind.append(True)

    dropped_idx =np.array(col_idxs)[np.array(drop_ind)]

    X_drp = X_orig.copy()
    X_drp[:,dropped_idx] = 0
    
    if scoring == "sketchy":
        perf_fnl = split_and_score(model, X, y, idx_splits)
    else:
        perf_fnl = cross_val_score(model, X_drp, y, scoring=scoring, cv=idx_splits, n_jobs=n_jobs).reshape(1,-1)

    perfs = np.concatenate(perfs, axis=0)
    drop_inds = np.array(drop_ind)
    col_idxs = np.array(col_idxs)
    
    summary_df = pd.DataFrame(
        {"col_idx": col_idxs,
         "perf_mean": perfs.mean(axis=1)
        })
    summary_df["best_perf_mean"] = summary_df["perf_mean"].cummax()
    summary_df["is_dropped"] = drop_inds
    summary_df["n_features"] = summary_df.shape[0] - summary_df["is_dropped"].cumsum()
    
    fnl_col_idx = list(summary_df["col_idx"][summary_df["is_dropped"]==False])

    return {"perf_metric": scoring,
            "perf_init": perf_init,
            "perf_fnl": perf_fnl,
            "perfs": perfs,
            "drop_inds": drop_inds,
            "col_idxs": col_idxs,
            "summary": summary_df,
            "fnl_col_idx": fnl_col_idx,
            "idx_splits": idx_splits}

def plot_dropcol(summary, figsize=[6,4], fontsize=12):
    fig, ax = plt.subplots(1,1, figsize=figsize)
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean Validation Accuracy", fontsize=fontsize)
    ax.plot(summary["perf_mean"], "o", color="blue")
    ax.plot(summary["best_perf_mean"], ".", color="orange")

    ax2 = ax.twinx()
    ax2.set_ylabel("Remaining Features", fontsize=fontsize)
    ax2.plot(summary["n_features"], ".", color="green")
    ax2.axis('tight')

    # add legend to distinguish series
    ax2.plot([], [], label="Current Perf")
    ax2.plot([], [], label="Best Perf")
    ax2.plot([], [], label="Remaining Features")
    ax2.legend(loc="upper left")
    
    return fig, ax