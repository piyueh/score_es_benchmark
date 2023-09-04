#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""Post processing for stats workflow performance benchmarks.
"""
#%%
import pathlib
import pstats
import itertools
import collections
import numpy
import matplotlib
import matplotlib.pyplot as pyplot
from postprocess_1 import fcolor, heatmap, annotate_heatmap

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

#%%
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

#%%
# create the figure
fig = pyplot.figure(figsize=(6, 3.79), dpi=166, layout="constrained")
fig.get_layout_engine().set(w_pad=0, h_pad=0, hspace=0, wspace=0)

gs = fig.add_gridspec(
    3, 2, width_ratios=[4.5, 5.5], height_ratios=[1, 1, 1],
    wspace=0.02, hspace=0.09
)

axs = [
    fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, :]),
    fig.add_subplot(gs[2, :]), fig.add_subplot(gs[0, 1])
]

# speedups
# ==============================================================================
names = ["Small", "Medium", "Large"]
values = numpy.zeros((2, 3), dtype=float)
values[0, :] = [data[("numpy", k.lower())]["total"] for k in names]
values[1, :] = [data[("c", k.lower())]["total"] for k in names]
values[1, :] = values[0, :] / values[1, :]
values[0, :] = 1.0

# normalizer mapping speedsup of segmental [0, 1] region
norm = matplotlib.colors.PowerNorm(0.2, vmin=1.0, vmax=50, clip=True)

# add the image to the axes
im = heatmap(
    values, ["NumPy", "C v4\nClang"], names, axs[0],
    cmap="coolwarm", norm=norm
)

# add annotations to the cell centers
texts = annotate_heatmap(im, values, "{x:.1f}x", fontsize=13, weight="bold")

# tuning
axs[0].tick_params(labelsize=13, labelrotation=0)
axs[0].set_title("Whole-Workflow Speedups", fontsize=14, weight="bold")

# times
# ==============================================================================
# bar section: qcf percentage
starts = numpy.zeros(len(names))
values = numpy.array([data[("c", k.lower())]["qcf"] for k in names])
axs[1].barh(names, values, 0.6, starts)

# bar section: cross section
starts += values
values = numpy.array([data[("c", k.lower())]["cross"] for k in names])
axs[1].barh(names, values, 0.6, starts)

# bar section: inverse CDF sampling
starts += values
values = numpy.array([data[("c", k.lower())]["sampling"] for k in names])
axs[1].barh(names, values, 0.6, starts)

# bar section: loss
starts += values
values = numpy.array([data[("c", k.lower())]["loss"] for k in names])
axs[1].barh(names, values, 0.6, starts)

# bar section: other
starts += values
values = numpy.array([data[("c", k.lower())]["other"] for k in names])
axs[1].barh(names, values, 0.6, starts)

# tune the lower plots
axs[1].set_title("Whole-Workflow Run Time (min)", fontsize=14, weight="bold")
axs[1].tick_params(bottom=False, labelbottom=False, top=True, labeltop=True)
axs[1].tick_params(labelsize=13)
axs[1].set_facecolor("whitesmoke")

# profiling
# ==============================================================================
# bar section: qcf percentage
arts = []
starts = numpy.zeros(len(names))
values = numpy.array([
    data[("c", k.lower())]["qcf"]/data[("c", k.lower())]["total"]*100
    for k in names
])
arts.append(axs[2].barh(names, values, 0.6, starts, label="QCF"))

# bar section: cross section
starts += values
values = numpy.array([
    data[("c", k.lower())]["cross"]/data[("c", k.lower())]["total"]*100
    for k in names
])
arts.append(axs[2].barh(names, values, 0.6, starts, label="Cross-Section"))

# bar section: inverse CDF sampling
starts += values
values = numpy.array([
    data[("c", k.lower())]["sampling"]/data[("c", k.lower())]["total"]*100
    for k in names
])
arts.append(axs[2].barh(names, values, 0.6, starts, label="Sampling"))

# bar section: loss
starts += values
values = numpy.array([
    data[("c", k.lower())]["loss"]/data[("c", k.lower())]["total"]*100
    for k in names
])
arts.append(axs[2].barh(names, values, 0.6, starts, label="Loss"))

# bar section: other
starts += values
values = numpy.array([
    data[("c", k.lower())]["other"]/data[("c", k.lower())]["total"]*100
    for k in names
])
arts.append(axs[2].barh(names, values, 0.6, starts, label="Others"))

# tune the lower plots
axs[2].tick_params(bottom=False, labelbottom=False, top=True, labeltop=True)
axs[2].tick_params(labelsize=13)
axs[2].set_xlim(0., 101.)
axs[2].set_title("Whole-Workflow Profiling (Percentage %)", fontsize=14, weight="bold")
axs[2].set_facecolor("whitesmoke")

# legends
# ==============================================================================
axs[3].axis("off")
axs[3].spines[:].set_visible(False)
# axs[3].text(
#     0.5, 0.94, "Problem Size\n(# of Events)",
#     transform=axs[3].transAxes, fontsize=14, va="bottom", ha="center",
#     ma="center", weight="bold"
# )
# axs[3].text(
#     0.5, 0.89, u"\u2022 Large: 50000\n\u2022 Medium: 10000\n\u2022 Small: 1000",
#     transform=axs[3].transAxes, fontsize=14, va="top", ha="center", ma="left",
#     linespacing=2.0, bbox=dict(boxstyle="square", fc='whitesmoke', ec="k")
# )
# axs[3].text(
#     0.5, 0.99, "Code Components",
#     transform=axs[3].transAxes, fontsize=13, va="top", ha="center",
#     ma="center"
# )
axs[3].set_title("Code Components", fontsize=13)
axs[3].legend(
    handles=arts, loc="upper center", bbox_to_anchor=(0.5, 1.1), ncols=2,
    fontsize=12, facecolor="whitesmoke", edgecolor="k", fancybox=False,
    markerscale=0.6, columnspacing=0.5, handletextpad=0.2, labelspacing=0.2
)




# legends
# ==============================================================================

fig.savefig(imgdir.joinpath("overall_speedups.png"), bbox_inches="tight")