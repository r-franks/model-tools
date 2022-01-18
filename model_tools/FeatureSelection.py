from tqdm import tqdm
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def split_and_score(model, X, y, idx_splits):
    train_perfs = []
    val_perfs = []
    
    for train_idx, test_idx in idx_splits:
        model.fit(X[train_idx,:], y[train_idx])
        train_perf = model.score(X[train_idx,:], y[train_idx])
        val_perf = model.score(X[test_idx,:], y[test_idx])
        
        train_perfs.append(train_perf)
        val_perfs.append(val_perf)
        
    return np.array(train_perfs).T.reshape(1,-1), np.array(val_perfs).T.reshape(1,-1)

def dropcol_iterate(model, X, y, n_splits=3, col_order=None, random_state=42):
    X_orig = X.copy()
    X = X.copy()

    idx_splits = [s for s in KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(X)]
    train_init, val_init = split_and_score(model, X, y, idx_splits)
    
    val_mean = np.mean(val_init)
    
    train_perfs = []
    val_perfs = []
    best_vals = []
    col_idxs = []
    drop_ind = []
    
    if col_order is not None:
        iterator = tqdm(col_order)
    else:
        iterator = tqdm(range(X.shape[1]))
    for col_idx in iterator:
        col_tmp = X[:,col_idx].copy()
        X[:,col_idx] = 0
        
        # idx_splits = [s for s in KFold(n_splits=n_splits, shuffle=True, random_state=random_state+col_idx).split(X)]
        train_perf, val_perf = split_and_score(model, X, y, idx_splits)
        
        col_idxs.append(col_idx)
        train_perfs.append(train_perf)
        val_perfs.append(val_perf)
        
        val_mean_new = np.mean(val_perf)
        
        best_vals.append(val_mean)
        
        if val_mean > val_mean_new:
            X[:,col_idx] = col_tmp
            drop_ind.append(False)
        else:
            val_mean = val_mean_new
            drop_ind.append(True)

    dropped_idx =np.array(col_idxs)[np.array(drop_ind)]

    X_drp = X_orig.copy()
    X_drp[:,dropped_idx] = 0

    train_fnl, val_fnl = split_and_score(model, X_drp, y, idx_splits)

    train_perfs = np.concatenate(train_perfs, axis=0)
    val_perfs = np.concatenate(val_perfs, axis=0)
    drop_inds = np.array(drop_ind)
    col_idxs = np.array(col_idxs)
    
    summary_df = pd.DataFrame(
        {"col_idx": col_idxs,
         "train_mean": train_perfs.mean(axis=1),
         "val_mean": val_perfs.mean(axis=1)
        })
    summary_df["best_val_mean"] = summary_df["val_mean"].cummax()
    summary_df["is_dropped"] = drop_inds
    summary_df["n_features"] = summary_df.shape[0] - summary_df["is_dropped"].cumsum()
    
    fnl_col_idx = list(summary_df["col_idx"][summary_df["is_dropped"]==False])

    return {"train_init": train_init,
            "val_init": val_init,
            "train_fnl": train_fnl,
            "val_fnl": val_fnl,
            "train_perfs": train_perfs,
            "val_perfs": val_perfs,
            "drop_inds": drop_inds,
            "col_idxs": col_idxs,
            "summary": summary_df,
            "fnl_col_idx": fnl_col_idx,
            "idx_splits": idx_splits}

def plot_dropcol(summary, figsize=[6,4], fontsize=12):
    fig, ax = plt.subplots(1,1, figsize=figsize)
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean Validation Accuracy", fontsize=fontsize)
    ax.plot(summary["val_mean"], "o", color="blue")
    ax.plot(summary["best_val_mean"], ".", color="orange")

    ax2 = ax.twinx()
    ax2.set_ylabel("Remaining Features", fontsize=fontsize)
    ax2.plot(summary["n_features"], ".", color="green")
    ax2.axis('tight')

    # add legend to distinguish series
    ax2.plot([], [], label="Best Val Perf")
    ax2.plot([], [], label="Current Val Perf")
    ax2.plot([], [], label="Remaining Features")
    ax2.legend(loc="upper left")
    
    return fig, ax