#! /usr/bin/env python3
# vim:fenc=utf-8

"""Original score_es using NumPy and SciPy
"""
import numpy
import scipy
import pylibraft.distance
import cupy
import cupyx.scipy.spatial
import torch
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


def score_es_original_cupy(preds, obsrvs):
    """Using CuPy.
    """

    assert preds.dtype == cupy.float32, "CuPy's cdist only supports float32"
    assert obsrvs.dtype == cupy.float32, "CuPy's cdist only supports float32"

    pred_size, pred_event = preds.shape
    assert obsrvs.shape[1] == pred_event, "Observations and events have different sizes"

    # Calculate score1
    score1 = cupy.mean(cupyx.scipy.spatial.distance.cdist(preds, obsrvs))

    # Calculate score2
    score2 = cupyx.scipy.spatial.distance.pdist(preds)
    score2 = cupy.sum(score2) / (pred_size * (pred_size - 1))

    return score1 - score2


def score_es_original_raft(preds, obsrvs):
    """Using RapidsAI's Raft library.
    """
    pred_size, pred_event = preds.shape
    assert obsrvs.shape[1] == pred_event, "Observations and events have different sizes"

    # pre-allocate memory
    out1 = cupy.zeros((pred_size, obsrvs.shape[0]), dtype=preds.dtype)
    out2 = cupy.zeros((pred_size, pred_size), dtype=preds.dtype)

    # Calculate score1
    pylibraft.distance.pairwise_distance(preds, obsrvs, out=out1)

    # Calculate score2
    pylibraft.distance.pairwise_distance(preds, preds, out=out2)

    score1 = out1.mean()
    score2 = out2.sum()
    score2 /= (2 * pred_size * (pred_size - 1))

    return score1 - score2


def score_es_original_torch(preds, obsrvs):
    """Using PyTorch.
    """
    pred_size, pred_event = preds.shape
    assert obsrvs.shape[1] == pred_event, "Observations and events have different sizes"

    # Calculate score1
    score1 = torch.cdist(preds, obsrvs, p=2.0).mean()  # = euclidean

    # Calculate score2
    score2 = torch.nn.functional.pdist(preds, p=2.0).sum()  # = euclidean
    score2 /= (pred_size * (pred_size - 1))

    # TODO: synchronization?
    return score1 - score2


@torch.jit.script
def score_es_original_torchscript(preds, obsrvs):
    """Using PyTorch's TorchScript.
    """
    pred_size, pred_event = preds.shape
    assert obsrvs.shape[1] == pred_event, "Observations and events have different sizes"

    # Calculate score1
    score1 = torch.cdist(preds, obsrvs, p=2.0).mean()  # = euclidean

    # Calculate score2
    score2 = torch.nn.functional.pdist(preds, p=2.0).sum()  # = euclidean
    score2 /= (pred_size * (pred_size - 1))

    return score1 - score2
