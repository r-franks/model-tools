import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def highlight_region_around_x(ax, x_target, xs, spacing=0.5, highlight_color="orange", alpha=0.5):
    eps = 1e-8

    diffs = xs - x_target

    try:
        min_pos_diff = np.min(diffs[diffs > eps])
    except:
        min_pos_diff = 0.0

    try:
        min_neg_diff = np.max(diffs[diffs < -eps])
    except:
        min_neg_diff = 0.0

    rng_pos = x_target + min_pos_diff*spacing
    rng_neg = x_target + min_neg_diff*spacing

    ax.axvspan(rng_neg, rng_pos, color=highlight_color, alpha=alpha)


##########################################################################################
#                             L1/L2 Regularization                                       # 
##########################################################################################
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
        highlight_region_around_x(ax, x_target=np.log10(model.C_[0]), xs=np.log10(model.Cs_), spacing=0.5, highlight_color=highlight_c, alpha=0.5)
    
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
        highlight_region_around_x(ax, x_target=np.log10(model.C_[0]), xs=np.log10(model.Cs_), spacing=0.5, highlight_color=highlight_c, alpha=0.5)
    
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

##########################################################################################
#                                   Elastic Net                                          # 
##########################################################################################
def plot_perf_vs_l1ratio(model, highlight_c="orange", cmap=cm.viridis, t=0, figsize=None, fontsize=None, ax=None):
    # create figure if absent
    if not ax:
        fig, ax = plt.subplots(1,1, figsize=figsize)
    else:
        if figsize:
            warnings.warn("ax provided, figsize not updated")

    min_score, max_score = model.scores_[1].min(), model.scores_[1].max()
    min_score = (1-t)*min_score + t*max_score
    ax.set_ylim([min_score, max_score])

    colors = np.linspace(0, 1.0, model.Cs)
    for col, c, series in zip(colors, np.log10(model.Cs_), model.scores_[1].mean(axis=0)):
        ax.plot(model.l1_ratios, series, '-o', label="{0:.4f}".format(c), color=cmap(col))
        
    if highlight_c:
        highlight_region_around_x(ax, x_target=model.l1_ratio_[0], xs=model.l1_ratios, spacing=0.5, highlight_color=highlight_c, alpha=0.5)

    ax.set_xlabel("l1 ratio", fontsize=fontsize)
    ax.set_ylabel(model.scoring, fontsize=fontsize)
    return ax

def plot_perf_vs_c(model, highlight_c="orange", cmap=cm.viridis, t=0, figsize=None, fontsize=None, ax=None):
    # create figure if absent
    if not ax:
        fig, ax = plt.subplots(1,1, figsize=figsize)
    else:
        if figsize:
            warnings.warn("ax provided, figsize not updated")

    min_score, max_score = model.scores_[1].min(), model.scores_[1].max()
    min_score = (1-t)*min_score + t*max_score
    ax.set_ylim([min_score, max_score])

    colors = np.linspace(0, 1.0, len(model.l1_ratios))
    for col, l1ratio, series in zip(colors, model.l1_ratios, model.scores_[1].mean(axis=0).T):
        ax.plot(np.log10(model.Cs_), series, '-o', label="{0:.4f}".format(l1ratio), color=cmap(col))
        
    if highlight_c:
        highlight_region_around_x(ax, x_target=np.log10(model.C_[0]), xs=np.log10(model.Cs_), spacing=0.5, highlight_color=highlight_c, alpha=0.5)

    ax.set_xlabel("log(C)", fontsize=fontsize)
    ax.set_ylabel(model.scoring, fontsize=fontsize)
    return ax

def plot_elnet_perf(model, highlight_c="orange", t=0.0, figsize=None, fontsize=None):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=figsize)
    fig.suptitle('Elastic Net Average Crossval Performance', fontsize=fontsize)
    plot_perf_vs_l1ratio(model, highlight_c=highlight_c, t=t, fontsize=fontsize, ax=ax1)
    plot_perf_vs_c(model, highlight_c=highlight_c, t=t, fontsize=fontsize, ax=ax2)
    return fig, (ax1, ax2)