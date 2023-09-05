#! /usr/bin/env python3
# vim:fenc=utf-8

"""Original score_es using NumPy and SciPy
"""
import numpy
import scipy
from numba import njit


def score_es_original_numpy(preds, obsrvs):
    """
    Computes the energy score between the predicted and observed events.

    Parameters
    ----------
    preds : numpy.ndarray
    A 2D array of predicted events.

    obsrvs : numpy.ndarray
    A 2D array of observed events.

    Returns
    -------
    float
    The discriminative score between the predicted and observed events.
    """
    obs_size, obs_event = obsrvs.shape
    pred_size, pred_event = preds.shape
    assert obs_event == pred_event, "Observations and events have different sizes"

    # Calculate score1
    diff_1 = numpy.linalg.norm(preds[:, None] - obsrvs, ord=2, axis=2)
    score1 = numpy.mean(diff_1)

    # Calculate score2
    if pred_size > 1:
        diff_2 = numpy.linalg.norm(preds[:, None] - preds, ord=2, axis=2)
        numpy.fill_diagonal(diff_2, 0)
        score2 = numpy.sum(diff_2) / (2 * pred_size * (pred_size - 1))
    else:
        score2 = 0.0

    return score1 - score2


def score_es_original_scipy(preds, obsrvs):
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


@njit(nogil=True, boundscheck=False)
def score_es_original_numba(preds, obsrvs):
    """Naive Numba impl. Credit: Niteya

    Notes
    -----
    We tried to mimic the original Python code as much as possible. However, the original NumPy's
    norm function accepts an argument `axis`, while Numba's own norm implementation does not.
    We therefore had to modified the code somehow.
    """
    obs_size, obs_event = obsrvs.shape
    pred_size, pred_event = preds.shape
    assert obs_event == pred_event, "The second dimensions do not match"

    score_1 = 0.0
    score_2 = 0.0
    for i in range(pred_size):
        for j in range(obs_size):
            score_1 += numpy.linalg.norm(preds[i] - obsrvs[j])

        for j in range(i+1, pred_size):
            score_2 += numpy.linalg.norm(preds[i] - preds[j])

    score_1 /= (obs_size * pred_size)
    score_2 /= (pred_size * (pred_size - 1))

    return score_1 - score_2
