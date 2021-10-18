import warnings
import numpy as np
import matplotlib.pyplot as plt

def plot_reg_path_coef(model, highlight_c="orange", figsize=None, fontsize=None, ax=None):
    
    # create figure if absent
    if not ax:
        fig, ax = plt.subplots(1,1, figsize=figsize)
    else:
        if figsize:
            warnings.warn("ax provided, figsize not updated")
        
    ax.plot(np.log10(model.Cs_), model.coefs_paths_[1].mean(axis=0), marker='o')
    ymin, ymax = ax.set_ylim()
    
    if highlight_c:
        # determine window around best C parameter to highlight
        logdiff = np.diff(np.log10(model.Cs_))[0]
        prev_c = np.log10(model.C_[0]) - 0.5*logdiff
        next_c = np.log10(model.C_[0]) + 0.5*logdiff
        ax.axvspan(prev_c, next_c, color=highlight_c, alpha=0.5)
    
    ax.set_xlabel('log(C)', fontsize=fontsize)
    ax.set_ylabel('Mean Coefficient', fontsize=fontsize)
    ax.axis('tight')
    return ax

def plot_reg_path_perf(model, highlight_c="orange", figsize=None, fontsize=None, ax=None):
    # create figure if absent
    if not ax:
        fig, ax = plt.subplots(1,1, figsize=figsize)
    else:
        if figsize:
            warnings.warn("ax provided, figsize not updated")
        
    ax.plot(np.log10(model.Cs_), model.scores_[1].mean(axis=0), marker='o')    
    ymin, ymax = ax.set_ylim()
    
    if highlight_c:
        # determine window around best C parameter to highlight
        logdiff = np.diff(np.log10(model.Cs_))[0]
        prev_c = np.log10(model.C_[0]) - 0.5*logdiff
        next_c = np.log10(model.C_[0]) + 0.5*logdiff
        ax.axvspan(prev_c, next_c, color=highlight_c, alpha=0.5)
    
    ax.set_xlabel('log(C)', fontsize=fontsize)
    ax.set_ylabel(model.scoring, fontsize=fontsize)
    ax.axis('tight')
    return ax

def plot_reg_path(model, highlight_c="orange", figsize=None, fontsize=None):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=figsize)
    fig.suptitle('Mean Logistic Regression Path Over Crossval Folds', fontsize=fontsize)
    plot_reg_path_coef(model, highlight_c=highlight_c, fontsize=fontsize, ax=ax1)
    plot_reg_path_perf(model, highlight_c=highlight_c, fontsize=fontsize, ax=ax2)
    return fig, (ax1, ax2)