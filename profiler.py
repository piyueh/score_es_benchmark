#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""Profiling driver.
"""
import sys
import pathlib
import cProfile
import pstats
import numpy

_rootdir = pathlib.Path(__file__).resolve().parent
_rootdir.joinpath("results").mkdir(exist_ok=True)

pstats.f8 = lambda x: f"{x:8.16f}"

def profiler(target, casename, shape=(10000, 1), n_rep=20, rootdir=_rootdir, stdout=False):
    """The profiling driver.
    """

    filename = rootdir.joinpath("results", f"{shape}_{casename}.prof")

    pr = cProfile.Profile()

    rng = numpy.random.default_rng(1)

    # warm-up
    preds = rng.random(shape, dtype=float)
    ans = rng.random(shape, dtype=float)
    target(preds, ans)

    # repetitions
    print(f"CASE: {shape}, {casename}", end="")
    sys.stdout.flush()
    for i in range(n_rep):

        print(f" {i}", end="")
        sys.stdout.flush()
        preds = rng.random(shape, dtype=float)
        ans = rng.random(shape, dtype=float)

        pr.enable()
        target(preds, ans)
        pr.disable()
    print()
    sys.stdout.flush()

    # save profiling result
    pr.dump_stats(filename)

    # test and print the profiling results from the file
    stats = pstats.Stats(str(filename))  # pstats only accepts str path

    if stdout:
        stats.strip_dirs().sort_stats("cumulative", "tottime").print_stats()

    # find the per-cal cumtime of the score_es implementation
    stats_profile = stats.get_stats_profile()

    for fn_name, fn_stats in stats_profile.func_profiles.items():
        if "score_es" in fn_name:
            res = fn_stats.percall_cumtime
            break
    else:
        raise RuntimeError(f"Could not find score_es in {filename}.")

    return res


if __name__ == "__main__":
    import impls
    import pandas

    pc_cum_t_key = "CUMTIME/CALL (ms)"
    shapes = [(1000, 1), (10000, 1), (10000, 2), (10000, 6), (10000, 12)]


    targets = {
        key: getattr(impls, f"score_es_{key}")
        for key in impls.options
    }

    reps = {key: 20 for key in impls.options}
    reps["original"] = 5
    reps["numba_v1"] = 5

    results = pandas.DataFrame(
        columns=pandas.MultiIndex.from_product([shapes, [pc_cum_t_key, "SPEEDUP"]]),
        index=pandas.Index(targets.keys(), dtype=str, name="IMPL"),
        dtype=float
    )

    for shape in shapes:
        for name, target in targets.items():

            results.loc[name, (shape, pc_cum_t_key)] = \
                profiler(target, name, shape, reps[name]) * 1e3

            results.loc[name, (shape, "SPEEDUP")] = \
                results.at["original", (shape, pc_cum_t_key)] / \
                results.at[name, (shape, pc_cum_t_key)]

            print(results.xs(shape, axis=1, level=0).loc[name])

            # update the csv promptly
            results.to_csv(_rootdir.joinpath("results", "table.csv"))

    print(results)
