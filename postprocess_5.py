#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""Post processing for ensemble analysis results.
"""
#%%
import itertools
import pathlib
import pickle
import yaml
import numpy
from matplotlib import pyplot

# folders
#%%
rootdir = pathlib.Path(__file__).expanduser().resolve().parent
resdir = rootdir.joinpath("results")
imgdir = rootdir.joinpath("images")

# make sure output folders exist
imgdir.mkdir(exist_ok=True)

# determine config file per the problem size
cfgfile = rootdir.joinpath("configs", f"config.uq.yaml")

# get config
with open(cfgfile, "r") as fp:
    cfg = yaml.load(fp, yaml.Loader)

# aliases for conveniences
nbootstraps = cfg["ensembles"]["nbootstraps"]
nrepeats = cfg["ensembles"]["nrepeats"]
truth = cfg["misc"]["true_params"]
nparams = len(truth)

# data holder
losses = numpy.full(nbootstraps, numpy.Inf, dtype=float)
params = numpy.full((nbootstraps, nparams), numpy.NaN, dtype=float)
counters = numpy.zeros(nbootstraps, dtype=int)

# grab all result files
filenames = resdir.glob("uq.*.*.dat")

# loop over files
for filename in filenames:
    # get the bootstrap index
    iboot = int(filename.suffixes[0].lstrip("."))

    # add 1 to the counter
    counters[iboot] += 1

    # open and load the data
    with open(filename, "rb") as fp:
        data = pickle.load(fp)
    
    # only record the data if it's bettern than the current
    if data.fun < losses[iboot]:
        losses[iboot] = data.fun
        params[iboot, :] = data.x

# sanity check
assert numpy.all(counters == nrepeats)
assert numpy.all(numpy.isfinite(losses))
assert not numpy.any(numpy.isnan(params))

# plot
# =============================================================================
fig = pyplot.figure(figsize=(6, 3.79), dpi=166, layout="constrained")
fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, hspace=0, wspace=0)
gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 1], height_ratios=[1, 10, 10])

axs = []
axs.append(fig.add_subplot(gs[1, 0]))
axs.append(fig.add_subplot(gs[1, 1], sharey=axs[-1]))
axs.append(fig.add_subplot(gs[1, 2], sharey=axs[-1]))
axs.append(fig.add_subplot(gs[2, 0]))
axs.append(fig.add_subplot(gs[2, 1], sharey=axs[-1]))
axs.append(fig.add_subplot(gs[2, 2], sharey=axs[-1]))
axs.append(fig.add_subplot(gs[0, :]))

coloriter = iter(pyplot.get_cmap("turbo")(numpy.linspace(0.05, 0.95, nparams)))
names = [r"$N_u$", r"$a_u$", r"$b_u$", r"$N_d$", r"$a_d$", r"$b_d$"]

arts = []
for i in range(nparams):
    _, _, patches = axs[i].hist(
        params[:, i], bins=12, density=False, color=next(coloriter), rwidth=0.9,
        label=names[i]
    )
    arts.append(patches[0])

    line = axs[i].axvline(truth[i], color="k", ls="--", lw=2, label="Truth")

    axs[i].set_facecolor("whitesmoke")
    axs[i].tick_params(labelsize=8)
    axs[i].tick_params(axis="x")  # , labelrotation=-20)

    if i != 0 and i != 3:
        axs[i].tick_params(axis="y", left=False, labelleft=False)

arts.append(line)

axs[-1].axis("off")
axs[-1].legend(
    handles=arts, loc="center", bbox_to_anchor=(0.5, 0.5), fontsize=10,
    facecolor="whitesmoke", fancybox=False, ncols=7, borderpad=0.1,
    markerscale=0.3, handletextpad=0.2, labelspacing=0.2
)

fig.supylabel("Occurrences", fontsize=12)
fig.supxlabel("QCF Parameter Values", fontsize=12)
fig.suptitle(
    "QCF Parameter Distrubutions from Ensemble Analysis",
    fontsize=14, weight="bold"
)
fig.savefig(imgdir.joinpath("ensemble_analysis.png"), bbox_inches="tight")