#! /usr/bin/env python3
# vim:fenc=utf-8

"""Profiling driver.
"""
import pathlib
import time
import itertools
import numpy
import cupy
import torch


# default to float64 in torch
torch.set_default_tensor_type(torch.DoubleTensor)


def run(key, func, shape):
    """Execute a kernel benchmark.

    Returns
    -------
    wall time in ms.
    """

    if "Raft" in key[1]:
        device = cupy.cuda.Device(cupy.cuda.runtime.getDevice())
        preds = cupy.random.uniform(-5., 5., shape, dtype="float64")
        obsrvs = cupy.random.uniform(-5., 5., shape, dtype="float64")

        device.synchronize()
        t = time.perf_counter_ns()

        func(preds, obsrvs)

        device.synchronize()
        t = (time.perf_counter_ns() - t) / 1e6  # ns->ms

    elif "Torch" in key[1]:
        kwargs = dict(dtype=torch.float64, device="cuda", requires_grad=False)
        preds = torch.zeros(shape, **kwargs)
        obsrvs = torch.zeros(shape, **kwargs)
        preds.uniform_(-5., 5.)
        obsrvs.uniform_(-5., 5.)

        torch.cuda.synchronize()
        t = time.perf_counter_ns()

        func(preds, obsrvs)

        torch.cuda.synchronize()
        t = (time.perf_counter_ns() - t) / 1e6  # ns->ms

    else:
        preds = numpy.random.uniform(-5., 5., shape).astype(numpy.float64)
        obsrvs = numpy.random.uniform(-5., 5., shape).astype(numpy.float64)

        t = time.perf_counter_ns()
        func(preds, obsrvs)
        t = (time.perf_counter_ns() - t) / 1e6  # ns->ms

    return t


if __name__ == "__main__":
    import impls

    # repeatitions
    reps = 20

    # desired implementations for this benchmark
    wanted = [
        ("Original", "SciPy"),
        ("Original", "Raft"),
        ("Original", "Torch"),
        ("Original", "TorchScript"),
        ("C v4", "Numba"),
        ("C v4", "GCC"),
        ("C v4", "Clang"),
        ("Original", "NumPy"),
    ]

    targets = {k: impls.options[k] for k in wanted}

    # target shapes
    shapes = [
        (1000, 1), (1000, 2), (1000, 6), (1000, 12),
        (5000, 1), (5000, 2), (5000, 6), (5000, 12),
        (10000, 1), (10000, 2), (10000, 6), (10000, 12),
    ]

    # folders and files
    rootdir = pathlib.Path(__file__).resolve().parent
    logfile = rootdir.joinpath("results", f"perf_log_v2_float64.csv")

    # init log file
    with open(logfile, "w") as fp:
        fp.write("IMPL,Backend,NEvents,EventSize,Iteration,Time\n")

    # run
    for (key, func), shape in itertools.product(targets.items(), shapes):

        if key == ("Original", "NumPy") and shape == (10000, 12):
            continue  # because of OOM

        print(f"Running {key} and {shape}")

        # warm up
        print(f"Warming up")
        run(key, func, shape)

        for it in range(reps):
            t = run(key, func, shape)

            # save time of each repeat
            with open(logfile, "a") as fp:
                fp.write(f"{key[0]},{key[1]},{shape[0]},{shape[1]},{it},{t}\n")

            print(f"Done job {key} and {shape} at iter={it}: {t} ms")
