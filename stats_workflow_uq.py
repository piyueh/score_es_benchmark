#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""Performance benchamrks for the whole stats workflow 1.
"""
import sys
import pickle
import pathlib
import yaml
import numpy
import scipy.optimize
import utilities
from mpi4py import MPI


def config_per_rank_print(rank):
    """Configure the stdout of the current process to have rank info in prefix.
    """
    org_write = sys.stdout.write

    def new_write(*args, **kwargs):
        if len(args[0].lstrip(" \n")) != 0:
            s = f"[Rank {rank}] "
            args = (s + args[0].replace("\n", f"\n{s}"),) + \
                tuple(_ for _ in args[1:])
        org_write(*args, **kwargs)
        sys.stdout.flush()  # force output immediately

    sys.stdout.write = new_write


def split_jobs(total, rank, nprocs, comm=None):
    """Split job IDs.

    Arguments
    ---------
    total : int
        Total number of jobs.
    rank : int
        The rank ID of the current MPI process.
    nprocs : int
        Total number of ranks in the current communicator.
    comm : MPI.Comm|None
        If not None and is an MPI.Comm, a sanity check will be performed.

    Returns
    -------
    ibg, ied : int
        The starting and ending job IDs of process-`rank`.
    """
    # obtain the bootstrap range that the current rank needs to run
    quotient, remainder = divmod(total, nprocs)
    ibg = rank * quotient + min(rank, remainder)
    ied = ibg + quotient + int(rank < remainder)

    # sanity check
    if comm is not None:
        ibg_debug = comm.allgather(ibg)
        ied_debug = comm.allgather(ied)
        assert ied_debug[-1] == total
        assert all(i == j for i, j in zip(ied_debug[:-1], ibg_debug[1:]))

    return ibg, ied


if __name__ == "__main__":

    # mpi info
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
    config_per_rank_print(rank)

    # folders
    rootdir = pathlib.Path(__file__).expanduser().resolve().parent
    resdir = rootdir.joinpath("results")

    # make sure output folders exist
    resdir.mkdir(exist_ok=True)

    # determine config file per the problem size
    cfgfile = rootdir.joinpath("configs", f"config.uq.yaml")

    # get config
    with open(cfgfile, "r") as fp:
        cfg = yaml.load(fp, yaml.Loader)
        cfg["data"] = [cfgfile.parent.joinpath(p) for p in cfg["data"]]

    # experimental data loader
    dataloader = utilities.DataLoader(cfg["data"])

    # aliases for conveniences
    nbootstraps = cfg["ensembles"]["nbootstraps"]
    nrepeats = cfg["ensembles"]["nrepeats"]

    # get jobs
    ibg, ied = split_jobs(nbootstraps, rank, size, MPI.COMM_WORLD)
    MPI.COMM_WORLD.Barrier()  # just to make the stdout look nice

    # leave a stdout msg about the exact soln
    print(f"Exact: {cfg['misc']['true_params']}")

    # loop over bootstraps
    for iboot in range(ibg, ied):
        obsrvs = dataloader.bootstrap(cfg["env"]["nobsrvs"])

        for rep in range(nrepeats):

            cfg["env"]["seed"] = iboot * (10**len(str(nrepeats))) + rep
            cfg["optimizer"]["seed"] = 2 * cfg["env"]["seed"]
            env = utilities.StatsWorkflowEnv(cfg["env"], obsrvs)
            optimizer = utilities.Optimizer(env, cfg["optimizer"])
            res = optimizer.run()
            print(f"{(iboot, rep)}; {res.x}; {res.fun}")

            with open(resdir.joinpath(f"uq.{iboot}.{rep}.dat"), "wb") as fp:
                pickle.dump(res, fp)

    # sanity check
    print(f"Done")
