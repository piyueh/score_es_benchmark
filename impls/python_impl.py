#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""Original score_es
"""
import numpy
import scipy


def score_es_original(predicted_events, observed_events):
    """
    Computes the energy score between the predicted and observed events.

    Parameters
    ----------
    predicted_events : numpy.ndarray
    A 2D array of predicted events.

    observed_events : numpy.ndarray
    A 2D array of observed events.

    Returns
    -------
    float
    The discriminative score between the predicted and observed events.
    """
    obs_size, obs_event = observed_events.shape
    pred_size, pred_event = predicted_events.shape
    assert obs_event == pred_event, "Observations and events have different sizes"

    # Calculate score1
    diff_1 = numpy.linalg.norm(predicted_events[:, None] - observed_events, ord=2, axis=2)
    score1 = numpy.mean(diff_1)

    # Calculate score2
    if pred_size > 1:
        diff_2 = numpy.linalg.norm(predicted_events[:, None] - predicted_events, ord=2, axis=2)
        numpy.fill_diagonal(diff_2, 0)
        score2 = numpy.sum(diff_2) / (2 * pred_size * (pred_size - 1))
    else:
        score2 = 0.0

    return score1 - score2


def score_es_scipy(preds, obsrvs):
    """The fallback of score_es when numba and c implementations are not available.
    """
    pred_size, pred_event = preds.shape
    assert obsrvs.shape[1] == pred_event, "Observations and events have different sizes"

    # Calculate score1
    score1 = numpy.mean(scipy.spatial.distance.cdist(preds, obsrvs))

    # Calculate score2
    score2 = scipy.spatial.distance.pdist(preds)
    score2 = numpy.sum(score2) / (pred_size * (pred_size - 1))

    return score1 - score2


if __name__ == "__main__":
    import sys
    import pathlib
    import importlib

    def _import_helper(module_name, filepath):
        """Import a module using its file path.

        Arguments
        ---------
        module_name : str
            The module name that will be registered in sys.modules.
        filepath : str or os.PathLike
            The path to the source file.

        Returns
        -------
        module : a Python module
        """
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    root = pathlib.Path(__file__).resolve().parent
    profiler = getattr(_import_helper("profiler", root.parent.joinpath("profiler.py")), "profiler")

    result = profiler(score_es_original, "score_es_original", n_rep=1)
    result = profiler(score_es_scipy, "score_es_scipy", n_rep=1)
