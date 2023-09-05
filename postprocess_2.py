#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""Post-processing for plotting loss surface
"""
import pathlib
import numpy
import matplotlib.pyplot as pyplot

rootdir = pathlib.Path(__file__).expanduser().resolve().parent
datafile = rootdir.joinpath("results", "earth_mover_surface.npy")
imgfile = rootdir.joinpath("images", "earth_mover_surface.png")
imgfile.parent.mkdir(parents=True, exist_ok=True)

# these parameters should match those in earth_mover_surface.py
max_disturb = 0.5

data = numpy.load(datafile)
data = data.mean(axis=0)

X, Y = numpy.meshgrid(
    numpy.linspace(0., max_disturb, data.shape[1]),
    numpy.linspace(0., max_disturb, data.shape[0])
)

fig, axs = pyplot.subplots(
    1, 2, sharex=True, figsize=(7, 3.2), dpi=166, layout="constrained")

im = axs[0].contour(X, Y, data, 8, cmap="turbo")
ctxt = axs[0].clabel(im, inline=True, fontsize=14)

for txt in ctxt:
    txt.set_c("k")
    txt.set_bbox(dict(facecolor="whitesmoke", edgecolor="none", pad=2))

axs[0].set_ylabel(r"Perturb. lvl. in $N_d$, $a_d$, & $b_d$", fontsize=14)

im = axs[1].contourf(X, Y, data, 16, cmap="turbo")
axs[1].tick_params(left=False, right=False, labelleft=False)

fig.supxlabel(r"Perturb. lvl. in $N_u$, $a_u$, & $b_u$", fontsize=14)

cbar = fig.colorbar(im, ax=axs)
cbar.set_label(label="Earth-Mover Loss Value", size=14)

pyplot.savefig(imgfile)
