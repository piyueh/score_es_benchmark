#! /usr/bin/env python3
# vim:fenc=utf-8

"""Generate data to mimic true experimental event data.
"""
import pathlib
import numpy
import yaml
import utilities

if __name__ == "__main__":

    # folder/file paths
    rootdir = pathlib.Path(__file__).expanduser().resolve().parent
    cfgfile = rootdir.joinpath("config.yaml")

    # read config
    with open(rootdir.joinpath("config.yaml"), "r") as fp:
        config = yaml.load(fp, yaml.Loader)

    # aliases
    nevents = config["misc"]["n_true_events"]
    params = config["misc"]["true_params"]

    # the theoritical model
    qcf = utilities.QCF(0.1, 0.99999)
    cross_1 = utilities.CrossSectionIMPL1(qcf)
    cross_2 = utilities.CrossSectionIMPL2(qcf)
    sampler_1 = utilities.Sampler(cross_1, npts=1000, seed=111)
    sampler_2 = utilities.Sampler(cross_2, npts=1000, seed=222)

    # sampling events
    type_1_events = sampler_1(nevents[0], params)
    type_2_events = sampler_2(nevents[1], params)

    # get theoretical norms for the two types of corss section
    norm_1 = cross_1.get_norm(params)
    norm_2 = cross_2.get_norm(params)

    # data file paths are relative to the parent of the config file
    path_1 = cfgfile.parent.joinpath(pathlib.Path(config["data"][0]))
    path_2 = cfgfile.parent.joinpath(pathlib.Path(config["data"][1]))

    # save data and create missing parent folders
    path_1.parent.mkdir(parents=True, exist_ok=True)
    path_2.parent.mkdir(parents=True, exist_ok=True)
    numpy.savez(path_1, events=type_1_events, norm=norm_1)
    numpy.savez(path_2, events=type_2_events, norm=norm_2)