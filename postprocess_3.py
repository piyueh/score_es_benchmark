#! /usr/bin/env python3
# vim:fenc=utf-8

"""Post processing for stats workflow performance benchmarks.
"""
import pathlib
import pstats
import itertools
import collections
import numpy
import matplotlib
import matplotlib.pyplot as pyplot

# default color cycler
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(
    color=pyplot.get_cmap("turbo")(numpy.linspace(0.07, 1., 5))
)

rootdir = pathlib.Path(__file__).expanduser().resolve().parent
resdir = rootdir.joinpath("results")
imgdir = rootdir.joinpath("images")
resdir.mkdir(exist_ok=True)
imgdir.mkdir(exist_ok=True)

impls = ["numpy", "c"]
pbsizes = ["small", "medium", "large"]

keys = ["step", "get_samples", "get_norm", "get_cross_section", "get_ud"]

losskey = dict(
    numpy="score_es_original_numpy",
    c="<built-in method impls.c_v4.c_v4_clang.score_es_c_v4>"
)

data = {}

for pbsize, impl in itertools.product(pbsizes, impls):

    print(f"Processing {pbsize} {impl} ...")

    try:
        stats = pstats.Stats(
            str(resdir.joinpath(f"profile.{pbsize}.{impl}.dat"))
        )
    except FileNotFoundError:
        print("\n\tNo Data\n")
        continue

    temp = collections.defaultdict(lambda: 0.)

    for key, val in stats.stats.items():

        if key[-1] in keys:
            temp[key[-1]] += val[3] / 1e9 / 60
        elif key[-1] == losskey[impl]:
            temp["loss"] += val[3] / 1e9 / 60
        else:
            continue  # redundant, just for clarity

    data[(impl, pbsize)] = dict(
        total=temp["step"],
        qcf=temp["get_ud"],
        cross=temp["get_cross_section"]-temp["get_ud"],
        sampling=temp["get_samples"]-temp["get_cross_section"],
        loss=temp["loss"],
        other=temp["step"]-temp["get_samples"]-temp["loss"]
    )

# create the figure
fig = pyplot.figure(figsize=(15, 4), dpi=166, layout="constrained")
fig.get_layout_engine().set(w_pad=0, h_pad=0, hspace=0, wspace=0)
gs = fig.add_gridspec(2, 3, width_ratios=[8, 1.3, 2], hspace=0.01)
axs = [
    fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[1, :2]),
    fig.add_subplot(gs[:, 2])
]

# use a ghost axes to add title for the top figure
topax = fig.add_subplot(gs[0, :2])
topax.axis('off')
topax.set_title("Run time (min)", fontsize=16, weight="bold")
topax.tick_params(
    bottom=False, labelbottom=False, top=True, labeltop=True, labelsize=15
)

# upper plots: run times
# ==============================================================================

# bar section: qcf time
labels = ["small", "medium", "large"]
names = ["Small", "Medium", "Large"]
starts = numpy.zeros(len(labels))
values = numpy.array([data[("numpy", k)]["qcf"] for k in labels])
axs[0].barh(names, values, 0.8, starts)
axs[1].barh(names, values, 0.8, starts)

# bar section: cross section
starts += values
values = numpy.array([data[("numpy", k)]["cross"] for k in labels])
axs[0].barh(names, values, 0.8, starts)
axs[1].barh(names, values, 0.8, starts)

# bar section: inverse CDF sampling
starts += values
values = numpy.array([data[("numpy", k)]["sampling"] for k in labels])
axs[0].barh(names, values, 0.8, starts)
axs[1].barh(names, values, 0.8, starts)

# bar section: loss
starts += values
values = numpy.array([data[("numpy", k)]["loss"] for k in labels])
axs[0].barh(names, values, 0.8, starts)
axs[1].barh(names, values, 0.8, starts)

# bar section: other
starts += values
values = numpy.array([data[("numpy", k)]["other"] for k in labels])
axs[0].barh(names, values, 0.8, starts)
axs[1].barh(names, values, 0.8, starts)

# plot axis broken symbols
axs[0].fill_betweenx(
    numpy.linspace(0., 1.0),
    1.0,
    numpy.sin(2.*numpy.linspace(0., 1.) * numpy.pi)*0.01+0.99,
    transform=axs[0].transAxes,
    color="w"
)
axs[1].fill_betweenx(
    numpy.linspace(0., 1.0),
    0.,
    numpy.sin(2.*numpy.linspace(0., 1.) * numpy.pi)*0.06+0.07,
    transform=axs[1].transAxes,
    color="w"
)

# additional broken axis symbols
axs[0].plot(
    [1, 1], [0, 1], ls="none", marker=[(-1, -1), (1, 1)], markersize=12,
    color="k", mec="k", mew=1, clip_on=False, transform=axs[0].transAxes
)
axs[1].plot(
    [0, 0], [0, 1], ls="none", marker=[(-1, -1), (1, 1)], markersize=12,
    color="k", mec="k", mew=1, clip_on=False, transform=axs[1].transAxes
)

# tuning the upper plots
axs[0].set_xlim(0., 15.)
axs[0].set_xticks(numpy.arange(0, 15, 2))
axs[0].tick_params(
    right=False, labelright=False, bottom=False, labelbottom=False,
    top=True, labeltop=True, labelsize=15
)
axs[0].spines["right"].set_visible(False)
axs[0].set_facecolor("whitesmoke")
axs[1].set_xlim(200, 202)
axs[1].tick_params(
    left=False, labelleft=False, bottom=False, labelbottom=False,
    top=True, labeltop=True, labelsize=15
)
axs[1].spines["left"].set_visible(False)
axs[1].set_facecolor("whitesmoke")
axs[1].set_xticks([201])

# lower plots: percentages
# ==============================================================================

# bar section: qcf percentage
arts = []
starts = numpy.zeros(len(labels))
values = numpy.array([
    data[("numpy", k)]["qcf"]/data[("numpy", k)]["total"]*100
    for k in labels
])
arts.append(axs[2].barh(names, values, 0.8, starts, label="QCF"))

# bar section: cross section
starts += values
values = numpy.array([
    data[("numpy", k)]["cross"]/data[("numpy", k)]["total"]*100
    for k in labels
])
arts.append(axs[2].barh(names, values, 0.8, starts, label="Cross-Section"))

# bar section: inverse CDF sampling
starts += values
values = numpy.array([
    data[("numpy", k)]["sampling"]/data[("numpy", k)]["total"]*100
    for k in labels
])
arts.append(axs[2].barh(names, values, 0.8, starts, label="Sampling"))

# bar section: loss
starts += values
values = numpy.array([
    data[("numpy", k)]["loss"]/data[("numpy", k)]["total"]*100
    for k in labels
])
arts.append(axs[2].barh(names, values, 0.8, starts, label="Loss"))

# bar section: other
starts += values
values = numpy.array([
    data[("numpy", k)]["other"]/data[("numpy", k)]["total"]*100
    for k in labels
])
arts.append(axs[2].barh(names, values, 0.8, starts, label="Others"))

# tune the lower plots
axs[2].set_xlim(0., 101.)
axs[2].set_title("Percentage (%)", fontsize=16, weight="bold")
axs[2].tick_params(
    bottom=False, labelbottom=False, top=True, labeltop=True, labelsize=15
)
axs[2].set_facecolor("whitesmoke")

# far right: legends, comments
# ==============================================================================
axs[3].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
axs[3].spines[:].set_visible(False)
axs[3].text(
    0.5, 0.94, "Problem Size\n(# of Events)",
    transform=axs[3].transAxes, fontsize=14, va="bottom", ha="center",
    ma="center", weight="bold"
)
axs[3].text(
    0.5, 0.89, u"\u2022 Large: 50000\n\u2022 Medium: 10000\n\u2022 Small: 1000",
    transform=axs[3].transAxes, fontsize=14, va="top", ha="center", ma="left",
    linespacing=2.0, bbox=dict(boxstyle="square", fc='whitesmoke', ec="k")
)
axs[3].text(
    0.5, 0.46, "Code Components",
    transform=axs[3].transAxes, fontsize=14, va="bottom", ha="center",
    ma="center", weight="bold"
)
axs[3].legend(
    handles=arts, loc="upper center", bbox_to_anchor=(0.5, 0.46),
    fontsize=14, facecolor="whitesmoke", edgecolor="k", fancybox=False,
)

fig.savefig(imgdir.joinpath("profiling_before.png"), bbox_inches="tight")