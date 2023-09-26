#! /usr/bin/env python3
# vim:fenc=utf-8

"""Misc. utilities w/ PyTorch.
"""
import pathlib
import numbers
import numpy
import scipy.integrate
import scipy.interpolate
import scipy.optimize
import torch
import impls


class Bootstraper:
    """An infinite iterator to bootstrapping data from datasets.

    It is basically a wrapper to a bunch of `torch.utils.data.DataLoader`. We
    use `DataLoader` because it provides multiprocessing pre-fetching and CUDA
    pinned memory mechanisms.

    Arguments
    ---------
    paths : List[str|os.PathLike]
        Paths to NumPy `npy` or `npz` datasets represents data from all kinds
        of experimental types.
    nevents : List[int]
        Numbers of samples to draw from datasets.
    num_workers : int
        See Torch's `DataLoader`. This class' default is `2`.
    pin_memory : bool
        See Torch's `DataLoader`. This class' default is `True`.
    **kwargs :
        All other arguments to `DataLoader`.
    """

    def __init__(
        self, paths, nevents, seeds=None, num_workers=2, pin_memory=True,
        **kwargs
    ):

        # pre-process input arguments
        if not hasattr(seeds, "__len__"):  # a scalar or None:
            seeds = [seeds for _ in len(paths)]

        self.paths = [pathlib.Path(p).expanduser().resolve() for p in paths]
        self.nevents = nevents
        self.seeds = seeds

        self.datasets = []
        self.norms = []
        self.loaders = []
        self.iters = []

        for p, n, s in zip(self.paths, self.nevents, self.seeds):

            _data = numpy.load(p)
            self.datasets.append(torch.from_numpy(_data["events"]))
            self.norms.append(torch.from_numpy(_data["norm"]))

            self.loaders.append(torch.utils.data.DataLoader(
                dataset=self.datasets[-1],
                batch_sampler=self.batch_idx_generator(self.datasets[-1], n, s),
                num_workers=num_workers, pin_memory=pin_memory, **kwargs
            ))

            self.iters.append(iter(self.loaders[-1]))

    def __iter__(self):  # infinity iterator
        while True:
            yield [next(it) for it in self.iters], self.norms

    def __call__(self, channels=None):
        """Get bootstraps from one or all datasets.

        Arguments
        ---------
        channels : int|List[int]|None
            Which datasets to draw samples. If `None` (default), draw from all
            datasets.

        Returns
        -------
        events : torch.Tensor|List[torch.Tensor]
            Sampled event data.
        norms : torch.Tensor|List[torch.Tensor]
            Norms of the corresponding datasets. A norm is a scalar but wrapped
            as a zero-dimensional tensor.
        """

        if channels is None:
            channels = range(len(self.datasets))

        try:
            return \
                [next(self.iters[ch]) for ch in channels], \
                [self.norms[ch] for ch in channels]
        except TypeError as err:
            if "not iterable" in str(err):
                return next(self.iters[channels]), self.norms[channels]
            raise

    @staticmethod
    def batch_idx_generator(data, nsamples, seed=None):
        ndata = len(data)
        rng = numpy.random.default_rng(seed)
        while True:
            yield rng.choice(ndata, nsamples, True).tolist()


class QCF:
    """A mock QCF.

    Arguments
    ---------
    xmin, xmax : float
        The lower and upper limit of allowed x values.
    """

    def __init__(self, xmin=1e-5, xmax=1.-1e-5):
        self.xmin = xmin
        self.xmax = xmax

    @staticmethod
    def get_u(x, params):
        """Get u.
        """
        u = params[0] * (x**params[1]) * ((1.0 - x)**params[2])
        return u

    @staticmethod
    def get_d(x, params):
        d = params[0] * (x**params[1]) * ((1.0 - x)**params[2])
        return d

    @staticmethod
    def get_ud(x, params):
        u = params[0] * (x**params[1]) * ((1.0 - x)**params[2])
        d = params[3] * (x**params[4]) * ((1.0 - x)**params[5])
        return u, d

    def __call__(self, x, params):
        return self.get_ud(x, params)

    def verify_x(self, x):
        """Check if values in an array x are all within the limits.
        """
        return numpy.logical_and(x >= xmin, x <= xmax)


class CrossSectionBase:
    """A base class for cross-section given a QCF model.

    Arguments
    ---------
    qcf : QCF
        An instance of class `QCF`.
    """

    def __init__(self, qcf: QCF):
        self.qcf = qcf  # a ref, not a hard copy
        self.xmin = self.qcf.xmin  # for convenience
        self.xmax = self.qcf.xmax  # for convenience
        self._quad = scipy.integrate.quad  # for convenience

        # caches
        self.cache_params = None
        self.cache_norm = None

    def get_cross_section(self, x, params):
        """Get cross-section at x with given QCF parameters.
        """
        raise NotImplementedError

    def __call__(self, x, params):
        return self.get_cross_section(x, params)

    def get_norm(self, params):
        """Get the norm of the cross-section with given QCF parameters.
        """
        params = numpy.asarray(params)  # zero-overhead if params is ndarray

        if not numpy.array_equal(params, self.cache_params):
            self.cache_params = params
            self.cache_norm = self._quad(
                self.get_cross_section, self.xmin, self.xmax, (params,))[0]

        return self.cache_norm


class CrossSectionIMPL1(CrossSectionBase):
    """A mock cross-section given a QCF model.

    Arguments
    ---------
    qcf : QCF
        An instance of class `QCF`.
    """

    def get_cross_section(self, x, params):
        u, d = self.qcf(x, params)
        return 4.0 * u + d


class CrossSectionIMPL2(CrossSectionBase):
    """A mock cross-section given a QCF model.

    Arguments
    ---------
    qcf : QCF
        An instance of class `QCF`.
    """

    def get_cross_section(self, x, params):
        u, d = self.qcf(x, params)
        return u + 4.0 * d


class Sampler:
    """A simple inverse CDF sampler.
    """

    def __init__(self, cross, npts=100, seed=None):
        self.cross = cross
        self.npts = npts
        self.xmin = self.cross.xmin  # for convenience
        self.xmax = self.cross.xmax  # for convenience
        self.x = torch.linspace(
            self.xmin, self.xmax, self.npts, dtype=torch.float64, device="cpu",
            requires_grad=False
        )
        self._quad = numpy.vectorize(
            scipy.integrate.quad, excluded={"b", 2, "args", 3}
        )
        self._rng = numpy.random.default_rng(seed)

    def pdf(self, x, params, norm=None):
        """Evaluate PDF at x with given QCF params.
        """
        if norm is None:
            norm = self.cross.get_norm(params)
        return self.cross(x, params) / norm

    def cdf(self, x, params):
        """Evaluate CDF at x with given QCF parameters.
        """
        norm = self.cross.get_norm(params)
        return self._quad(self.pdf, x, self.xmax, (params, norm))[0]

    def get_icdf_func(self, params):
        """Get a callable object representing the inverse CDF function.
        """
        return scipy.interpolate.interp1d(
            self.cdf(self.x, params), self.x,
            bounds_error=False, fill_value=0
        )

    def icdf(self, y, params):
        """Evaluate inverse CDF values at y with given QCF parameters.

        This function: probability density \mapsto x
        """
        icdf_func = self.get_icdf_func(params)
        return icdf_func(y)

    def get_samples(self, nevents, params):
        """Get samples from the underlying cross-section with given QCF params.
        """
        samples = self.icdf(self._rng.uniform(0., 1., nevents), params)
        samples = samples.reshape(-1, 1)
        return samples

    def __call__(self, nevents, params):
        return self.get_samples(nevents, params)


class StatsLoss:
    """Loss function for stats workflow 1 using NumPy.
    """

    def __init__(self, weights=1., impl=("C v4", "Clang")):

        # make aliases to frequently used settings for conveniences
        self.impl = tuple(impl)

        if isinstance(weights, numbers.Number):
            self.weights = numpy.full(2, weights, dtype=float)
        else:
            self.weights = numpy.asarray(weights, dtype=float)

        # temporary data holders for logging purpose outside this class
        self.cost = 0.
        self.costs = numpy.zeros((2,), dtype=float)
        self.weighted_costs = numpy.zeros((2,), dtype=float)

        # actual implementation of the earth-mover loss
        self.score_es = impls.options[self.impl]

    def forward(self, preds: list, obsrvs: list) -> float:
        """Calculate the total loss.

        Arguments
        ---------
        obsrvs : list[dict(events: ndarray, norm: float)]
            Its length should be the number of channels. Each element in it is a
            dictionary of two key-value pairs:
            1) `events`: an array of shape (n_events, event_size) for observed
               events of the corresponding data channel;
            2) `norm`: float-like; the norm of the observed events.
        preds : list[dict(events: ndarray, norm: float)]
            The predicted events. `preds` has the same structure as `obsrvs`.
            However, the array in `events` can have different number of rows.

        Returns
        -------
        cost : float or a numpy.floating
            Weighted and aggregated loss.
        """

        self.costs[0] = self.score_es(preds["events"], obsrvs["events"])
        self.costs[1] = (obsrvs["norm"] - preds["norm"])**2

        self.weighted_costs[...] = self.weights * self.costs
        self.cost = numpy.sum(self.weighted_costs)

        return self.cost


class StatsWorkflowEnv:
    """Mock environment.
    """
    def __init__(self, config, obsrvs):

        self.obsrvs = obsrvs
        self.nevents = config["nevents"]
        self.seed = config["seed"]

        # underlying QCFs
        self.qcf = QCF(**config["qcf"])

        # corss-sections
        self.crosses = [
            CrossSectionIMPL1(self.qcf), CrossSectionIMPL2(self.qcf)
        ]

        # samplers for different cross-sections
        self.samplers = [
            Sampler(_1, **_2)
            for _1, _2 in zip(self.crosses, config["samplers"])
        ]

        # loss functions for different cross-sections
        self.lossfns = [StatsLoss(**_1) for _1 in config["losses"]]

        # sanity check
        assert len(self.crosses) == len(self.samplers)
        assert len(self.samplers) == len(self.lossfns)
        assert len(self.lossfns) == len(self.nevents)
        assert len(self.nevents) == len(self.obsrvs)

    def step(self, params):
        """Stepping function.
        """

        iters = enumerate(zip(
            self.crosses, self.samplers, self.lossfns, self.nevents,
            self.obsrvs
        ))

        total = 0.
        for i, (cross, sampler, lossfn, nevents, obsrvs) in iters:

            # each env always uses the same sample points from inverse CDF
            sampler._rng = numpy.random.default_rng(self.seed**i)

            preds = {
                "events": sampler.get_samples(nevents, params),
                "norm": cross.get_norm(params),
            }

            loss = lossfn.forward(preds, obsrvs)
            total += loss

        return total


class Optimizer:
    """A mock optimizer.
    """
    def __init__(self, env, config):
        self.env = env
        self.options = config["options"]
        self.parmin = config["parmin"]
        self.parmax = config["parmax"]
        self.seed = config["seed"]
        self.rng = numpy.random.default_rng(self.seed)
        self.bounds = [(_1, _2) for _1, _2 in zip(self.parmin, self.parmax)]

    def run(self):
        """Run.

        Returns
        -------
        An instance of `scipy.optimize.OptimizeResult`
        """
        x0 = self.rng.uniform(self.parmin, self.parmax)

        res: scipy.optimize.OptimizeResult = scipy.optimize.minimize(
            self.env.step, x0,
            method="L-BFGS-B", jac="2-point",
            options=self.options, bounds=self.bounds
        )

        return res
