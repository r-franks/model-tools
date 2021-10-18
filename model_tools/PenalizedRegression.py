import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors

def zero_one_normalize(xs):
    return (xs - xs.min())/(xs.max() - xs.min())

def plot_cbar(xs, cmap, fig, ax, label=None, fontsize=None):
    normalize = mcolors.Normalize(vmin=xs.min(), vmax=xs.max())
    scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=cmap)
    scalarmappaple.set_array(xs)
    cbar = fig.colorbar(scalarmappaple, ax=ax)
    cbar.set_label(label, fontsize=fontsize)
    return cbar

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
def plot_reg_path_coef(model, marker='o', highlight_c="orange", figsize=None, fontsize=None, ax=None):
    # create figure if absent
    return_fig = False
    if not ax:
        fig, ax = plt.subplots(1,1, figsize=figsize)
        return_fig = True
    else:
        if figsize:
            warnings.warn("ax provided, figsize not updated")
        
    ax.plot(np.log10(model.Cs_), model.coefs_paths_[1].mean(axis=0), marker=marker)
    ymin, ymax = ax.set_ylim()
    
    if highlight_c:
        highlight_region_around_x(ax, x_target=np.log10(model.C_[0]), xs=np.log10(model.Cs_), spacing=0.5, highlight_color=highlight_c, alpha=0.5)
    
    ax.set_xlabel('log(C)', fontsize=fontsize)
    ax.set_ylabel('Mean Coefficient', fontsize=fontsize)
    ax.axis('tight')

    if return_fig:
        return fig, ax
    else:
        return ax

def plot_reg_path_perf(model, marker='o', highlight_c="orange", include_n_coef=False, figsize=None, fontsize=None, ax=None):
    # create figure if absent
    return_fig = False
    if not ax:
        fig, ax = plt.subplots(1,1, figsize=figsize)
        return_fig = True
    else:
        if figsize:
            warnings.warn("ax provided, figsize not updated")
        
    ax.plot(np.log10(model.Cs_), model.scores_[1].mean(axis=0), label=model.scoring, marker=marker)    
    ymin, ymax = ax.set_ylim()
    
    if include_n_coef:
        ax_coef = ax.twinx()
        ax_coef.set_ylabel("features", fontsize=fontsize)
        ax_coef.plot(np.log10(model.Cs_), (model.coefs_paths_[1] != 0).any(axis=0).sum(axis=1), marker=marker, label="n coef", color="orange")
        ax_coef.axis('tight')

        # add legend to distinguish series
        ax.plot([], [], label="n coef")
        ax.legend()

    if highlight_c:
        highlight_region_around_x(ax, x_target=np.log10(model.C_[0]), xs=np.log10(model.Cs_), spacing=0.5, highlight_color=highlight_c, alpha=0.5)
    
    ax.set_xlabel('log(C)', fontsize=fontsize)
    ax.set_ylabel(model.scoring, fontsize=fontsize)
    ax.axis('tight')

    if return_fig:
        return fig, ax
    else:
        return ax

def plot_reg_path(model, marker='o', highlight_c="orange", include_n_coef=False, figsize=None, fontsize=None):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=figsize)
    fig.suptitle('Mean Logistic Regression Path Over Crossval Folds', fontsize=fontsize)
    plot_reg_path_coef(model, marker=marker, highlight_c=highlight_c, fontsize=fontsize, ax=ax1)
    plot_reg_path_perf(model, marker=marker, highlight_c=highlight_c, include_n_coef=include_n_coef, fontsize=fontsize, ax=ax2)
    return fig, (ax1, ax2)

##########################################################################################
#                                   Elastic Net                                          # 
##########################################################################################
def plot_perf_vs_l1ratio(model, marker='o', highlight_c="orange", cmap=cm.viridis, t=0, figsize=None, fontsize=None, ax=None):
    # create figure if absent
    return_fig = False
    if not ax:
        fig, ax = plt.subplots(1,1, figsize=figsize)
        return_fig = True
    else:
        if figsize:
            warnings.warn("ax provided, figsize not updated")

    min_score, max_score = model.scores_[1].min(), model.scores_[1].max()
    min_score = (1-t)*min_score + t*max_score
    ax.set_ylim([min_score, max_score])

    log10Cs = np.log10(model.Cs_)
    colors = zero_one_normalize(log10Cs)
    for col, c, series in zip(colors, log10Cs, model.scores_[1].mean(axis=0)):
        ax.plot(model.l1_ratios, series, marker=marker, label="{0:.4f}".format(c), color=cmap(col))
        
    if highlight_c:
        highlight_region_around_x(ax, x_target=model.l1_ratio_[0], xs=model.l1_ratios, spacing=0.5, highlight_color=highlight_c, alpha=0.5)

    ax.set_xlabel("l1 ratio", fontsize=fontsize)
    ax.set_ylabel(model.scoring, fontsize=fontsize)

    if return_fig:
        plot_cbar(log10Cs, cmap=cmap, fig=fig, ax=ax, label="log(C)", fontsize=fontsize)
        return fig, ax
    else:
        return ax

def plot_perf_vs_c(model, marker='o', highlight_c="orange", cmap=cm.viridis, t=0, figsize=None, fontsize=None, ax=None):
    # create figure if absent
    return_fig = False
    if not ax:
        fig, ax = plt.subplots(1,1, figsize=figsize)
        return_fig = True
    else:
        if figsize:
            warnings.warn("ax provided, figsize not updated")

    min_score, max_score = model.scores_[1].min(), model.scores_[1].max()
    min_score = (1-t)*min_score + t*max_score
    ax.set_ylim([min_score, max_score])

    colors = zero_one_normalize(model.l1_ratios)
    for col, l1ratio, series in zip(colors, model.l1_ratios, model.scores_[1].mean(axis=0).T):
        ax.plot(np.log10(model.Cs_), series, marker=marker, label="{0:.4f}".format(l1ratio), color=cmap(col))
        
    if highlight_c:
        highlight_region_around_x(ax, x_target=np.log10(model.C_[0]), xs=np.log10(model.Cs_), spacing=0.5, highlight_color=highlight_c, alpha=0.5)

    ax.set_xlabel("log(C)", fontsize=fontsize)
    ax.set_ylabel(model.scoring, fontsize=fontsize)

    if return_fig:
        plot_cbar(model.l1_ratios, cmap=cmap, fig=fig, ax=ax, label="l1 ratio", fontsize=fontsize)
        return fig, ax
    else:
        return ax

def plot_elnet_perf(model, marker='o', highlight_c="orange", cmap=cm.viridis, t=0, figsize=None, fontsize=None):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=figsize)
    fig.suptitle('Elastic Net Average Crossval Performance', fontsize=fontsize)
    plot_perf_vs_l1ratio(model, marker=marker, highlight_c=highlight_c, cmap=cmap, t=t, fontsize=fontsize, ax=ax1)
    plot_perf_vs_c(model, marker=marker, highlight_c=highlight_c, cmap=cmap, t=t, fontsize=fontsize, ax=ax2)

    plot_cbar(np.log10(model.Cs_), cmap=cmap, fig=fig, ax=ax1, label="log(C)", fontsize=fontsize)
    plot_cbar(model.l1_ratios, cmap=cmap, fig=fig, ax=ax2, label="l1 ratio", fontsize=fontsize)

    return fig, (ax1, ax2)