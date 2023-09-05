#! /usr/bin/env python3
# vim:fenc=utf-8

"""Profiling driver.
"""
import pathlib
import time
import itertools
import numpy


if __name__ == "__main__":
    from mpi4py import MPI
    from impls import options  # available kernels

    # alias: rank info
    rank = MPI.COMM_WORLD.rank
    nprocs = MPI.COMM_WORLD.size

    # setup this rank's name
    name = f"Rank {rank:02d}"

    # folders and files
    rootdir = pathlib.Path(__file__).resolve().parent
    logfile = rootdir.joinpath("results", f"log.csv.{rank:02d}")

    # create required folders
    if rank == 0:
        logfile.parent.mkdir(exist_ok=True)
    MPI.COMM_WORLD.Barrier()

    # target shapes
    shapes = [
        (1000, 1), (1000, 2), (1000, 6), (1000, 12),
        (5000, 1), (5000, 2), (5000, 6), (5000, 12),
        (10000, 1), (10000, 2), (10000, 6), (10000, 12)
    ]

    # calculate job iteration ranges
    reps = 1000
    quotient, remainder = divmod(reps, nprocs)
    bg = rank * quotient +  (rank if rank <= remainder else remainder)
    ed = bg + quotient + int(rank < remainder)
    print(f"[{name}] quotient={quotient}, remainder={remainder}")
    print(f"[{name}] bg={bg}, ed={ed}")

    # init log file
    with open(logfile, "w") as fp:
        fp.write("IMPL,Backend,NEvents,EventSize,Iteration,Time\n")

    # run
    for (key, func), shape in itertools.product(options.items(), shapes):

        print(f"[{name}] running {key} and {shape} from {bg} to {ed}")

        # warm up
        print(f"[{name}] warming up")
        preds = numpy.random.uniform(-5., 5., shape)
        obsrvs = numpy.random.uniform(-5., 5., shape)
        func(preds, obsrvs)

        for it in range(bg, ed):
            preds = numpy.random.uniform(-5., 5., shape)
            obsrvs = numpy.random.uniform(-5., 5., shape)
            t = time.perf_counter_ns()
            func(preds, obsrvs)
            t = (time.perf_counter_ns() - t) / 1e6  # ns->ms

            # save time of each repeat
            with open(logfile, "a") as fp:
                fp.write(f"{key[0]},{key[1]},{shape[0]},{shape[1]},{it},{t}\n")

            print(f"[{name}] done job {key} and {shape} at iter={it}: {t} ms")
