#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""Post-processing for the speedups of the loss function alone.
"""
#%%
import pathlib
import re
import pandas
import numpy
import matplotlib
import matplotlib.pyplot as pyplot


def fcolor(x):
    """Calculate foreground color according to a given background color.

    Arguments
    ---------
    x : sequence-alike
        RGB representation of the given background color.

    Returns
    -------
    RGB representation of the foreground color.
    """

    luminance = 0.2126 * x[0] + 0.7152 * x[1] + 0.0722 * x[2]
    return (1., 1., 1.) if (luminance < 140./256.) else (0., 0., 0.)


def heatmap(data, row_labels, col_labels, ax, **kwargs):
    """Create a heatmap from a numpy array and two lists of labels.

    Modified from Matplotlib Gallery: Creating Annotated Heatmaps.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted. If
        not provided, use current axes or create a new one.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    # Plot the heatmap
    im = ax.imshow(data, aspect="auto", **kwargs)

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(numpy.arange(data.shape[1]))
    ax.set_yticks(numpy.arange(data.shape[0]))
    ax.set_xticklabels(col_labels, rotation=50)
    ax.set_yticklabels(row_labels)

    # Disable the short bars of ticks
    ax.tick_params(top=False, bottom=False, left=False, right=False)

    # put tick labels at bottom and left
    ax.tick_params(labelleft=True, labelbottom=True)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(numpy.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(numpy.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def annotate_heatmap(
    im, data, valfmt, textcolors=("black", "white"),
    threshold=None, **textkw
):
    """A function to annotate a heatmap.

    Modified from Matplotlib Gallery: Creating Annotated Heatmaps.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.
    valfmt
        The format str of the annotations inside the heatmap.
    textcolors
        A pair of colors, for values below and above a threshold, respectively.
    threshold
        The threshold used by `textcolors`.
    **kwargs
        All other arguments forwarded to the `axes.text` method.
    """

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be overwritten by textkw
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter
    valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=fcolor(im.cmap(im.norm(data[i, j]))[:3]))
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


#%%
if __name__ == "__main__":

    rootdir = pathlib.Path(__file__).expanduser().resolve().parent
    datadir = rootdir.joinpath("results")
    imgdir = rootdir.joinpath("images")
    imgdir.mkdir(exist_ok=True)

    # col names
    colnames = ["IMPL", "Backend", "NEvents", "EventSize", "Time"]

    # data type of each column
    coltypes = {
        "IMPL": str, "Backend": str, "NEvents": int, "EventSize": int,
        "Iteration": int, "Time": float  # Iteration not used
    }

    # search data files
    datafiles = [
        fp for fp in datadir.iterdir()
        if re.match(r"^log\.csv\.\d+$", fp.name) is not None
    ]

    # initialize an empty data holder
    times = pandas.DataFrame(columns=colnames)

    # loop through each data file and merge to the data holder
    for fp in datafiles:
        times = times.merge(
            right=pandas.read_csv(fp, usecols=colnames, dtype=coltypes),
            how="outer",
            on=colnames,
            sort=True,
        )

    # calculate mean of each group
    times = times.groupby(["IMPL", "Backend", "NEvents", "EventSize"])
    times = times.mean()

    # flatten out the nested indices
    times = times.reset_index()

    # change the table's layout
    times = times.pivot(
        columns=["NEvents", "EventSize"],
        index=["IMPL", "Backend"],
        values="Time"
    )

    # remove index and column names for better visualization
    times.index.names = ["", ""]
    times.columns.names = ["", ""]

    # calculate speedups and round to only 1 decimal
    speedups = times.loc[("Original", "NumPy")] / times
    speedups = speedups.round(1)

    print(times)
    print(speedups)

    # aliases for convenience
    ncols = len(speedups.columns)
    nrows = len(speedups.index)
    collabels = list([f"{col}" for col in speedups.columns])
    rowlabels = list([f"{col[0]}, {col[1]}" for col in speedups.index])
    data = speedups.values

    #%%
    # normalizer mapping speedsup of segmental [0, 1] region
    norm = matplotlib.colors.PowerNorm(0.5, vmin=1.0, vmax=150, clip=True)

    # create a figure and an axes in it
    fig, ax = pyplot.subplots(1, 1, dpi=166, figsize=(12, 5.05), tight_layout=True)

    # add the image to the axes
    im = heatmap(data, rowlabels, collabels, ax, cmap="coolwarm", norm=norm)

    # add annotations to the cell centers
    texts = annotate_heatmap(im, data, "{x:.1f}x", weight="bold")

    ax.set_title("Speedups of the Loss Function Alone", fontsize=18, weight="bold")
    ax.tick_params(labelsize=13, axis="y")
    ax.tick_params(labelsize=13, labelrotation=30, axis="x")

    fig.savefig(imgdir.joinpath("earth_mover_speedups.png"), dpi=166)
