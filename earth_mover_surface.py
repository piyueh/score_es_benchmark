#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""Plot the surface of the Earth-Mover loss.
"""
import sys
import argparse
import pathlib
import multiprocessing
import itertools
import yaml
import numpy
import scipy.integrate
import matplotlib.pyplot as pyplot
import utilities

def worker(
    inpq: multiprocessing.JoinableQueue,
    outq: multiprocessing.Queue,
    seed,
    cfg,
    nevents,
):
    """A worker process.
    """
    name = multiprocessing.current_process().name
    print(f"[{name}] started")


    truth = cfg["misc"]["true_params"]  # true parameters
    dataloader = utilities.DataLoader(cfg["data"], seed=seed+13579)
    rng = numpy.random.default_rng(seed=seed)
    params = numpy.zeros_like(truth)

    # initialize theory, sampler, and the loss
    qcf = utilities.QCF()

    sigmas = [utilities.CrossSectionIMPL1(qcf), utilities.CrossSectionIMPL2(qcf)]
    samplers = [
        utilities.Sampler(sigma, seed=seed*10)
        for sigma in sigmas
    ]
    losses = [utilities.StatsLoss([1., 0.]) for _ in range(2)]

    while True:
        try:
            trial, yj, xi, x_dist, y_dist = inpq.get(block=True, timeout=10)
        except multiprocessing.queues.Empty:
            inpq.close()
            outq.close()
            print(f"[{name}] done")
            break

        params[:3] = truth[:3] + rng.normal(0., x_dist, 3)
        params[3:] = truth[3:] + rng.normal(0., y_dist, 3)

        # bootstrap true data
        obsrvs = dataloader.bootstrap(nevents)

        # get prediction
        preds = [
            {
                "events": sampler(n, params),
                "norm": sampler.cross.get_norm(params)
            }
            for sampler, n in zip(samplers, nevents)
        ]

        # loss
        val = sum([
            loss.forward(pred, obsrv)
            for loss, pred, obsrv in zip(losses, preds, obsrvs)
        ])

        outq.put((trial, yj, xi, val), True)
        inpq.task_done()
        print(f"[{name}] {(trial, yj, xi)} done; val={val}")
    return

def dispatcher(ntrials, nx, ny, xdists, ydists):
    """Job dispatcher.
    """
    name = multiprocessing.current_process().name
    print(f"[{name}] started")
    for trial, xi, yj in itertools.product(range(ntrials), range(nx), range(ny)):
        inpq.put((trial, yj, xi, xdists[xi], ydists[yj]), block=True)
        print(f"[{name}] put {(trial, yj, xi)}: {xdists[xi]}, {ydists[yj]}")
    inpq.close()
    print(f"[{name}] done")
    return

def collector(outq, ntrials, nx, ny, outdir):
    """Result collector.
    """
    name = multiprocessing.current_process().name
    print(f"[{name}] started")

    # result holder
    results = numpy.zeros((ntrials, ny, nx), dtype=float)

    counter = 0
    while True:
        try:
            trial, yj, xi, val = outq.get(block=True, timeout=10)
        except multiprocessing.queues.Empty:
            outq.close()
            print(f"[{name}] done")
            break

        results[trial, yj, xi] = val
        print(f"[{name}] collect {(trial, yj, xi)}: {val}")

        counter += 1
        if counter % (nx * ny) == 0:
            numpy.save(outdir.joinpath("earth_mover_surface.npy"), results)
            sys.stdout.flush()

    # final save
    numpy.save(outdir.joinpath("earth_mover_surface.npy"), results)
    print(results)

    return


if __name__ == "__main__":

    # cmd args
    parser = argparse.ArgumentParser()
    parser.add_argument("nprocs", type=int)
    nprocs = parser.parse_args().nprocs

    # folders
    rootdir = pathlib.Path(__file__).expanduser().resolve().parent
    cfgdir = rootdir.joinpath("configs")
    resultdir = rootdir.joinpath("results")
    resultdir.mkdir(exist_ok=True)

    # read true parameters from config file
    with open(cfgdir.joinpath("config.surface.yaml"), "r") as fp:
        cfg = yaml.load(fp, yaml.Loader)
        cfg["data"] = [cfgdir.joinpath(pathlib.Path(p)) for p in cfg["data"]]

    # configurations specific to this loss-surface plot
    max_disturb = 0.5
    nx = 50
    ny = 50
    ntrials = 400
    nevents = [5000, 5000]
    rng = numpy.random.default_rng()

    # calculate the disturbance level of each grid point
    x_disturb = numpy.linspace(0., max_disturb, nx)
    y_disturb = numpy.linspace(0., max_disturb, ny)

    # create input and output queues
    inpq = multiprocessing.JoinableQueue()
    outq = multiprocessing.Queue()

    # dispatch job
    print("[Master] launching dispatcher")
    dispatch_job = multiprocessing.Process(
        target=dispatcher, name="Dispatcher",
        args=(ntrials, nx, ny, x_disturb, y_disturb)
    )
    dispatch_job.start()

    # allocate workers
    print("[Master] allocating workers")
    for rank in range(nprocs):
        proc = multiprocessing.Process(
            target=worker, name=f"Worker {rank:02d}",
            args=(inpq, outq, rng.integers(0, 99999), cfg, nevents)
        )
        proc.start()

    # collecting job
    print("[Master] launching collector")
    collect_job = multiprocessing.Process(
        target=collector, name="Collector",
        args=(outq, ntrials, nx, ny, resultdir)
    )
    collect_job.start()

    # wait
    dispatch_job.join()
    inpq.join()
    collect_job.join()

    # close queues
    inpq.close()
    outq.close()
