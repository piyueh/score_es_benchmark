#! /usr/bin/env python3
# vim:fenc=utf-8

"""Performance benchamrks for the whole stats workflow 1.
"""
import argparse
import pathlib
import cProfile
import time
import yaml
import numpy
import utilities


if __name__ == "__main__":

    # cmd parser and arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "size", action="store", choices=["small", "medium", "large"], type=str,
        metavar="PROB_SIZE",
        help="The target problem size (\"small\", \"mediem\", \"large\")."
    )
    parser.add_argument(
        "impl", action="store", choices=["numpy", "c"], type=str,
        metavar="IMPL", help="Implementation. \"c\" or \"numpy\"."
    )
    args = parser.parse_args()

    # folders
    rootdir = pathlib.Path(__file__).expanduser().resolve().parent
    resultdir = rootdir.joinpath("results")

    # make sure output folders exist
    resultdir.mkdir(exist_ok=True)

    # determine config file per the problem size
    cfgfile = rootdir.joinpath("configs", f"config.{args.size}.{args.impl}.yaml")

    # get config
    with open(cfgfile, "r") as fp:
        cfg = yaml.load(fp, yaml.Loader)
        cfg["data"] = [cfgfile.parent.joinpath(p) for p in cfg["data"]]

    # experimental data loader
    dataloader = utilities.DataLoader(cfg["data"])

    env = utilities.StatsWorkflowEnv(
        config=cfg["env"],
        obsrvs=dataloader.bootstrap(cfg["env"]["nobsrvs"]),
    )

    optimizer = utilities.Optimizer(env, cfg["optimizer"])

    # start training and profiling
    pr = cProfile.Profile(time.perf_counter_ns, 1)  # measure in nano-seconds
    pr.enable()
    res = optimizer.run()
    pr.disable()
    pr.dump_stats(resultdir.joinpath(f"profile.{args.size}.{args.impl}.dat"))

    # sanity check
    assert res.success
    print(f"Exact: {cfg['misc']['true_params']}")
    print(f"Trained: {res.x}")
