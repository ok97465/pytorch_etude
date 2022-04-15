"""Helper for ML."""
# Standard library imports
from itertools import product
from typing import Optional

# Third party imports
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.pyplot import figure, subplots
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import arange, ndarray, newaxis


def _extents(f: ndarray) -> list[float]:
    """Extents of view of imagesc.

    Args:
      f: value list of axis

    Returns:
      List[float]: mean value of start and end of axis

    """
    delta = f[1] - f[0]
    return [f[0] - delta / 2, f[-1] + delta / 2]


def imagesc(
    *arg,
    cmap: str = "viridis",
    aspect: str = "auto",
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    colorbar: bool = False,
    colorbar_label: str = "",
    font_size: int = 10,
    fontweight: str = "normal",
    tick_font_size: Optional[int] = None,
    is_grid: bool = False,
    ax: Optional[Axes] = None,
    fig: Optional[Figure] = None,
) -> tuple[Figure, Axes]:
    """Draw image like imagesc in MATLAB.

    Args:
        *arg: DESCRIPTION.
        cmap: colormap, 'Greys', 'Greys_r', 'viridis', 'jet'. Defaults to "viridis".
        aspect: 'auto, 'equal'. Defaults to "auto".
        title: title of figure. Defaults to "".
        xlabel: xlabel. Defaults to "".
        ylabel: ylabel. Defaults to "".
        colorbar: enable colorbar. Defaults to False.
        colorbar_label: colorbar label. Defaults to "".
        font_size: font size. Defaults to 10.
        fontweight: 'normal', 'bold', 'light', 'medium', 'semibold', 'heavy', 'black'.
            Defaults to "normal".
        tick_font_size: tick. Defaults to None.
        is_grid: enable grid. Defaults to False.
        ax: Axes. Defaults to None.
        fig: Figure. Defaults to None.

    Returns:
        Figure, Axes

    """
    if len(arg) == 1:
        data = arg[0]
        nr = data.shape[0]
        nc = data.shape[1]
        col_axis_tmp = arange(nc)
        row_axis_tmp = arange(nr)
    elif len(arg) == 3:
        col_axis_tmp: ndarray = arg[0]
        row_axis_tmp: ndarray = arg[1]
        data = arg[2]
        if len(col_axis_tmp) != data.shape[1] or len(row_axis_tmp) != data.shape[0]:
            print("Plz Check length of axis variable of imagesc")
            return
    else:
        print("Plz Check argument of imagesc")
        return

    col_axis = _extents(col_axis_tmp)
    row_axis = _extents(row_axis_tmp)
    row_axis[1], row_axis[0] = row_axis[0], row_axis[1]

    if ax is None:
        fig, ax = subplots()

    im = ax.imshow(
        data,
        aspect=aspect,
        interpolation="none",
        extent=col_axis + row_axis,
        origin="upper",
        cmap=cmap,
    )
    ax.set_title(title, fontsize=font_size, fontweight=fontweight)
    ax.set_xlabel(xlabel, fontsize=font_size, fontweight=fontweight)
    ax.set_ylabel(ylabel, fontsize=font_size, fontweight=fontweight)

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=tick_font_size)

        if isinstance(colorbar_label, str) and colorbar_label != "":
            cbar.set_label(colorbar_label, fontsize=font_size, fontweight=fontweight)

    ax.tick_params(labelsize=tick_font_size)

    ax.grid(is_grid)

    if fig:
        fig.tight_layout()

    return fig, ax


def plot_confusion_matrix(
    cm: ndarray,
    labels: list,
    title: str = "",
    cm_normalize: bool = False,
    cmap: str = "Blues",
    cmap_log_scale: bool = False,
    figsize: Optional[tuple[float, float]] = None,
) -> tuple[Figure, Axes]:
    """Plot confusion matrix.

    Ref. https://deeplizard.com/learn/video/0LhiS6yu2qQ

    Args:
        cm: Confusion matrix.
        labels: Labels of classes.
        title: Figure title. Defaults to "".
        cm_normalize: Normailize confusion matrix. Defaults to False.
        cmap: Colormap name. Defaults to "Blues".
        cmap_log_scale: Use of a log color scale. Defaults to False.
        figsize: Size of figure. Defualts to None.

    Returns:
        tuple[Figure, Axes]: fig, ax1.

    """
    if cm_normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, newaxis]

    imshow_norm = None
    if cmap_log_scale:
        imshow_norm = LogNorm()

    fig = figure(figsize=figsize)
    ax1 = fig.add_subplot(1, 1, 1)

    im = ax1.imshow(cm, interpolation="nearest", cmap=cmap, norm=imshow_norm)

    ax1.set_title(title)
    ax1.grid(False)

    tick_marks = arange(len(labels))
    ax1.set_xticks(tick_marks, labels, rotation=45)
    ax1.set_yticks(tick_marks, labels)

    fig.colorbar(im)
    fig.tight_layout()

    fmt = ".2f" if cm_normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        ax1.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    return fig, ax1
