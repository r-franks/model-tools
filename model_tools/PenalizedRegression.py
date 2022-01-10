import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
import sklearn

# Designed for sklearn version 0.24.1

def zero_one_normalize(xs):
    """Rescales dataset features to [0,1] range
    
    Args: 
        xs: Pandas DataFrame
    """
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
#                            L1/L2 Regularization (logistic)                             # 
##########################################################################################
def check_object(LogisticRegressionCV):
    """ Checks that LogisticRegressionCV is fit with l1/l2 penalty
    """

    assert LogisticRegressionCV.penalty in ("l1", "l2"), "penalty must be l1 or l2"
    sklearn.utils.validation.check_is_fitted(LogisticRegressionCV) 

def plot_reg_path_coef(model, marker='o', highlight_c="orange", figsize=None, fontsize=None, ax=None):
    """Plots coefs vs penalization strength.
    
    Parameters
    ----------
    model: sklearn.linear_model.LogisticRegressionCV instance
        A fit LogisticRegressionCV with L1/L2 penalty for which plots are made

    marker: matplotlib.markers format, default='o'
        Marker type used in plots

    highlight_c: matplotlib color format or None, default="orange"
        If not None, the best penalization strength is highlighted by a bar of
        color highlight_c.

    figsize: tuple or list of floats or None, default=None
        Specifies the figure size for both plots combined.
        
    fontsize: int or None, default=None
        Specifies the font size used in labels and titles.
        
    ax: matplotlib.axes or None, default=None
        ax object to plot onto. If None, new ax is created.
        
    Returns
    -------
    ax (if ax!=None): matplotlib.axes
    fig, ax (if ax=None): matplotlib.pyplot.figure, matplotlib.axes
    """

    check_object(model)
    
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
    """Plots perf vs penalization strength with num of nonzero coefs if specified.
    
    Parameters
    ----------
    model: sklearn.linear_model.LogisticRegressionCV instance
        A fit LogisticRegressionCV with L1/L2 penalty for which plots are made

    marker: matplotlib.markers format, default='o'
        Marker type used in plots

    highlight_c: matplotlib color format or None, default="orange"
        If not None, the best penalization strength is highlighted by a bar of
        color highlight_c.
        
    include_n_coef: bool, default=False
        If true, the second plot also includes the number of nonzero 
        coefficients vs penalization strength on a second axis on the right.
    
    figsize: tuple or list of floats or None, default=None
        Specifies the figure size for both plots combined.
        
    fontsize: int or None, default=None
        Specifies the font size used in labels and titles.
        
    ax: matplotlib.axes or None, default=None
        ax object to plot onto. If None, new ax is created.
        
    Returns
    -------
    ax (if ax!=None): matplotlib.axes
    fig, ax (if ax=None): matplotlib.pyplot.figure, matplotlib.axes
    """
    
    check_object(model)
    
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
        ax_coef.set_ylabel("n coef", fontsize=fontsize)
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
    """Plots path of an L1/L2 regularized sklearn.linear_model.LogisticRegressionCV.
    Produces two adjacent plots.
    The first is a plot of mean coefficient values vs penalization strength.
    The second is a plot of performance vs penalization strength.
    The second plot may include number of nonzero coefs, if specified by parameters.
    
    Parameters
    ----------
    model: sklearn.linear_model.LogisticRegressionCV instance
        A fit LogisticRegressionCV with L1/L2 penalty for which plots are made

    marker: matplotlib.markers format, default='o'
        Marker type used in plots

    highlight_c: matplotlib color format or None, default="orange"
        If not None, the best penalization strength is highlighted by a bar of
        color highlight_c.
        
    include_n_coef: bool, default=False
        If true, the second plot also includes the number of nonzero 
        coefficients vs penalization strength on a second axis on the right.
    
    figsize: tuple or list of floats or None, default=None
        Specifies the figure size for both plots combined.
        
    fontsize: int or None, default=None
        Specifies the font size used in labels and titles.
    
    Returns
    -------
    fig, (ax1, ax2): matplotlib.pyplot.figure and matplotlib.axes for plots.
    """
    
    # Validate object
    check_object(model)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=figsize)
    # Create title
    fig.suptitle('Mean Logistic Regression Path Over Crossval Folds', fontsize=fontsize)
    # First plot
    plot_reg_path_coef(model, marker=marker, highlight_c=highlight_c, fontsize=fontsize, ax=ax1)
    # Second plot
    plot_reg_path_perf(model, marker=marker, highlight_c=highlight_c, include_n_coef=include_n_coef, fontsize=fontsize, ax=ax2)
    return fig, (ax1, ax2)

##########################################################################################
#                               Elastic Net (logistic)                                   # 
##########################################################################################
def plot_perf_vs_l1ratio(model, marker='o', highlight_c="orange", cmap=cm.viridis, t=None, figsize=None, fontsize=None, ax=None):
    """Plots path of Elastic Net sklearn.linear_model.LogisticRegressionCV.
    Performance is plotted vs penalization strength. 
    
    Parameters
    ----------
    model: sklearn.linear_model.LogisticRegressionCV instance
        A fit LogisticRegressionCV with L1/L2 penalty for which plots are made

    marker: matplotlib.markers format, default='o'
        Marker type used in plots

    highlight_c: matplotlib color format or None, default="orange"
        If not None, the best penalization strength is highlighted by a bar of
        color highlight_c.
        
    cmap: matplotlib colormap, default=matplotlib.cm.viridis
        Color map for series colors associated with penalization strength
        (first plot)
        
    t: int or None, default=None
        Defines lowest plotted performance by linear interpolation between
        lowest and highest performance. 
        t=None => Lowest plotted performance is worst performance above 0.5
        t=0 => Lowest plotted performance is the worst performance
        t=0.5 => Lowest plotted performance is halfway between worst and best
        t=1.0 => Lowest plotted performance is best performance
    
    figsize: tuple or list of floats or None, default=None
        Specifies the figure size for both plots combined.
        
    fontsize: int or None, default=None
        Specifies the font size used in labels and titles.
    
    Returns
    -------
    ax (if ax!=None): matplotlib.axes
    fig, ax (if ax=None): matplotlib.pyplot.figure, matplotlib.axes
    """  

    # create figure if absent
    return_fig = False
    if not ax:
        fig, ax = plt.subplots(1,1, figsize=figsize)
        return_fig = True
    else:
        if figsize:
            warnings.warn("ax provided, figsize not updated")

    # calculate mean crossval perfs
    mean_scores = model.scores_[1].mean(axis=0)
    max_score = mean_scores.max()
    
    if t:
        min_score = mean_scores.min()
        min_score = (1-t)*min_score + t*max_score
        ax.set_ylim([min_score, max_score])
    else:
        min_score = model.scores_[1].flatten()[model.scores_[1].flatten() > 0.5].min()
        ax.set_ylim([min_score, max_score])
        
    log10Cs = np.log10(model.Cs_)
    colors = zero_one_normalize(log10Cs)
    for col, c, series in zip(colors, log10Cs, mean_scores):
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

def plot_perf_vs_c(model, marker='o', highlight_c="orange", cmap=cm.viridis, t=None, figsize=None, fontsize=None, ax=None):
    """Plots path of Elastic Net sklearn.linear_model.LogisticRegressionCV.
    Performance is plotted vs l1 ratio. 
    
    Parameters
    ----------
    model: sklearn.linear_model.LogisticRegressionCV instance
        A fit LogisticRegressionCV with L1/L2 penalty for which plots are made

    marker: matplotlib.markers format, default='o'
        Marker type used in plots

    highlight_c: matplotlib color format or None, default="orange"
        If not None, the best penalization strength is highlighted by a bar of
        color highlight_c.
        
    cmap: matplotlib colormap, default=matplotlib.cm.viridis
        Color map for series colors associated with penalization strength
        (first plot)

    t: int or None, default=None
        Defines lowest plotted performance by linear interpolation between
        lowest and highest performance. 
        t=None => Lowest plotted performance is worst performance without 0.5
        t=0 => Lowest plotted performance is the worst performance
        t=0.5 => Lowest plotted performance is halfway between worst and best
        t=1.0 => Lowest plotted performance is best performance
    
    figsize: tuple or list of floats or None, default=None
        Specifies the figure size for both plots combined.
        
    fontsize: int or None, default=None
        Specifies the font size used in labels and titles.
    
    Returns
    -------
    ax (if ax!=None): matplotlib.axes
    fig, ax (if ax=None): matplotlib.pyplot.figure, matplotlib.axes
    """  

    # create figure if absent
    return_fig = False
    if not ax:
        fig, ax = plt.subplots(1,1, figsize=figsize)
        return_fig = True
    else:
        if figsize:
            warnings.warn("ax provided, figsize not updated")

    # calculate mean crossval perfs
    mean_scores = model.scores_[1].mean(axis=0)
    max_score = mean_scores.max()
    
    if t:
        min_score = mean_scores.min()
        min_score = (1-t)*min_score + t*max_score
        ax.set_ylim([min_score, max_score])
    else:
        min_score = model.scores_[1].flatten()[model.scores_[1].flatten() > 0.5].min()
        ax.set_ylim([min_score, max_score])

    colors = zero_one_normalize(model.l1_ratios)
    for col, l1ratio, series in zip(colors, model.l1_ratios, mean_scores.T):
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

def plot_elnet_perf(model, marker='o', highlight_c="orange", cmap1=cm.viridis, cmap2=cm.magma, t=0, figsize=None, fontsize=None):
    """Plots path of Elastic Net sklearn.linear_model.LogisticRegressionCV.
    Produces two adjacent plots.
    The first is a plot of performance vs l1 ratio.
    The second is a plot of performance vs penalization strength.
    
    Parameters
    ----------
    model: sklearn.linear_model.LogisticRegressionCV instance
        A fit LogisticRegressionCV with L1/L2 penalty for which plots are made

    marker: matplotlib.markers format, default='o'
        Marker type used in plots

    highlight_c: matplotlib color format or None, default="orange"
        If not None, the best penalization strength is highlighted by a bar of
        color highlight_c.
        
    cmap1: matplotlib colormap, default=matplotlib.cm.viridis
        Color map for series colors associated with penalization strength
        (first plot)
        
    cmap2: matplotlib colormap, default=matplotlib.cm.magma
        Color map for series colors associated with l1 ratio
        (second plot)
        
    t: int or None, default=None
        Defines lowest plotted performance by linear interpolation between
        lowest and highest performance. 
        t=None => Lowest plotted performance is worst performance without 0.5
        t=0 => Lowest plotted performance is the worst performance
        t=0.5 => Lowest plotted performance is halfway between worst and best
        t=1.0 => Lowest plotted performance is best performance
    
    figsize: tuple or list of floats or None, default=None
        Specifies the figure size for both plots combined.
        
    fontsize: int or None, default=None
        Specifies the font size used in labels and titles.
    
    Returns
    -------
    fig, (ax1, ax2): matplotlib.pyplot.figure and matplotlib.axes for plots.
    """    

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=figsize)
    fig.suptitle('Elastic Net Average Crossval Performance', fontsize=fontsize)
    plot_perf_vs_l1ratio(model, marker=marker, highlight_c=highlight_c, cmap=cmap1, t=t, fontsize=fontsize, ax=ax1)
    plot_perf_vs_c(model, marker=marker, highlight_c=highlight_c, cmap=cmap2, t=t, fontsize=fontsize, ax=ax2)

    plot_cbar(np.log10(model.Cs_), cmap=cmap1, fig=fig, ax=ax1, label="log(C)", fontsize=fontsize)
    plot_cbar(model.l1_ratios, cmap=cmap2, fig=fig, ax=ax2, label="l1 ratio", fontsize=fontsize)

    return fig, (ax1, ax2)

##########################################################################################
#                                 General (logistic)                                     # 
##########################################################################################
def plot_logistic_cv(model, marker=".", highlight_c="orange", figsize=None, fontsize=None):
    """Plots paths sklearn.linear_model.LogisticRegressionCV.
    
    Parameters
    ----------
    model: sklearn.linear_model.LogisticRegressionCV instance
        A fit LogisticRegressionCV with L1/L2 penalty for which plots are made

    marker: matplotlib.markers format, default='o'
        Marker type used in plots

    highlight_c: matplotlib color format or None, default="orange"
        If not None, the best penalization strength is highlighted by a bar of
        color highlight_c.
    
    figsize: tuple or list of floats or None, default=None
        Specifies the figure size for both plots combined.
        
    fontsize: int or None, default=None
        Specifies the font size used in labels and titles.
    
    Returns
    -------
    fig, (ax1, ax2): matplotlib.pyplot.figure and matplotlib.axes for plots.
    """  

    if model.penalty == "l1":
        return plot_reg_path(model, marker=marker, highlight_c=highlight_c, include_n_coef=True, figsize=figsize, fontsize=fontsize)
    elif model.penalty == "l2":
        return plot_reg_path(model, marker=marker, highlight_c=highlight_c, include_n_coef=False, figsize=figsize, fontsize=fontsize)
    elif model.penalty in ("elasticnet"):
        return plot_elnet_perf(model, marker=marker, highlight_c=highlight_c, figsize=figsize, fontsize=fontsize)
    else:
        raise ValueError("penalty must be l1, l2 or elasticnet")