""" Estimates multiple Covariance matrices (C_00, C_01, C_11) at the same time from running blocks
of data. These can the be used to perform cross-validated VAMP scoring of the input space.

"""

import numpy as np

from pyemma._base.fixed_seed import FixedSeedMixIn
from pyemma._base.serialization.serialization import SerializableMixIn
from pyemma.coordinates.data._base.streaming_estimator import StreamingEstimator
from pyemma._base.progress import ProgressReporter
from pyemma._ext.variational.estimators.running_moments import running_covar

import random


class _ShuffleSplit(object):

    def __init__(self, n_splits=3):
        self.n_splits = n_splits

    def split(self, covs):
        n = len(covs)
        full = np.arange(n)
        for _ in range(self.n_splits):
            train = np.random.choice(full, int(n / 2), replace=False)
            test = np.setdiff1d(full, train)
            yield train, test


def _worker(train, test, scoring_method, k, return_singular_values):
    assert _covs_global is not None
    traincovs = [_covs_global[i] for i in train]
    testcovs = [_covs_global[i] for i in test]
    return Covariances._score_impl(testcovs, traincovs, scoring_method, k, return_singular_values)


_covs_global = None


class Covariances(StreamingEstimator, SerializableMixIn, FixedSeedMixIn):
    __serialize_version = 0
    __serialize_fields = ('covs_', 'n_covs_')
    """
    Parameters
    ----------

    tau: int, default=5000
        size of running blocks of input stream.

    mode: str, default="sliding"

    """

    def __init__(self, n_covs, n_save=5, tau=5000, shift=10, stride=1, mode='sliding', assign_to_covs='random',
                 fixed_seed=False):
        super(Covariances, self).__init__()
        if mode not in ('sliding', 'linear'):
            raise ValueError('unsupported mode: %s' % mode)
        self.set_params(mode=mode, tau=tau, shift=shift, n_covs=n_covs, n_save=n_save,
                        assign_to_covs=assign_to_covs,
                        stride=stride, fixed_seed=fixed_seed)

    class _LinearCovariancesSplit(object):

        def __init__(self, block_size, shift, stride):
            self.block_size = block_size
            self.shift = shift
            self.stride = stride

        def n_chunks(self, iterable):
            return iterable.n_chunks(self.block_size, skip=self.shift, stride=self.stride) - 1

        def split(self, iterable):
            with iterable.iterator(chunk=self.block_size, return_trajindex=False, skip=self.shift,
                                   stride=self.stride) as it:
                current_chunk, next_chunk = None, None

                for current_data in it:
                    next_chunk = current_data

                    if current_chunk is not None:
                        yield current_chunk[:len(next_chunk)], next_chunk

                    if not it.last_chunk_in_traj:
                        current_chunk = next_chunk
                    else:
                        current_chunk, next_chunk = None, None

    class _SlidingCovariancesSplit(object):
        def __init__(self, block_size, offset, stride):
            self.block_size = block_size
            self.offset = offset
            self.stride = stride

        def n_chunks(self, iterable):
            n1 = iterable.n_chunks(2 * self.block_size - 1, stride=self.stride, skip=self.offset)
            n2 = iterable.n_chunks(2 * self.block_size - 1, stride=self.stride, skip=self.offset + self.block_size)
            return min(n1, n2)

        def split(self, iterable):
            with iterable.iterator(lag=self.block_size, chunk=2 * self.block_size - 1, return_trajindex=False,
                                   skip=self.offset, stride=self.stride) as it:
                for first_chunk, lagged_chunk in it:
                    yield first_chunk[:len(lagged_chunk)], lagged_chunk

    def _estimate(self, iterable):
        self.covs_ = np.array([running_covar(xx=True, xy=True, yy=True, remove_mean=True,
                                             symmetrize=False, sparse_mode='auto',
                                             modify_data=False, nsave=self.n_save) for _ in range(self.n_covs)])

        if self.mode == 'sliding':
            splitter = self._SlidingCovariancesSplit(self.tau, self.shift, self.stride)
        elif self.mode == 'linear':
            splitter = self._LinearCovariancesSplit(self.tau, self.shift, self.stride)
        else:
            raise NotImplementedError("unsupported mode: %s" % self.mode)

        pg = ProgressReporter()
        pg.register(splitter.n_chunks(iterable), "calculate covariances", 0)

        if self.assign_to_covs == 'round_robin':
            idx = 0

            def index():
                nonlocal idx
                res = idx % len(self.covs_)
                idx += 1
                return res

        elif self.assign_to_covs == 'random':
            random.seed(self.fixed_seed)

            def index():
                i = random.randint(0, len(self.covs_) - 1)
                return i
        else:
            raise NotImplementedError('unknown assign_to_covs mode: %s' % self.assign_to_covs)
        samples = np.zeros(self.n_covs, dtype=int)
        with pg.context(stage=0):
            for X, Y in splitter.split(iterable):
                index_ = index()
                self.covs_[index_].add(X, Y)
                pg.update(1, stage=0)
                samples[index_] += len(X)

        self.covs_ = np.array(list(filter(lambda c: len(c.storage_XX.storage) > 0, self.covs_)))
        self.samples_ = samples
        self.n_covs_ = len(self.covs_)

        if len(self.covs_) != self.n_covs_:
            self.logger.info("truncated covariance matrices due to lack of data (%s -> %s)", len(self.covs_),
                             self.n_covs_)

        return self

    @staticmethod
    def _aggregate(covs, bessel=True):
        old_weights_xx = [c.weight_XX() for c in covs]
        old_weights_xy = [c.weight_XY() for c in covs]
        old_weights_yy = [c.weight_YY() for c in covs]
        cumulative_weight_xx = sum(old_weights_xx)
        cumulative_weight_xy = sum(old_weights_xy)
        cumulative_weight_yy = sum(old_weights_yy)
        for c in covs:
            if len(c.storage_XX.storage) > 0:
                c.storage_XX.moments.w = cumulative_weight_xx
            if len(c.storage_XY.storage) > 0:
                c.storage_XY.moments.w = cumulative_weight_xy
            if len(c.storage_YY.storage) > 0:
                c.storage_YY.moments.w = cumulative_weight_yy
        c00 = sum(c.cov_XX(bessel=bessel) for c in covs)
        c01 = sum(c.cov_XY(bessel=bessel) for c in covs)
        c11 = sum(c.cov_YY(bessel=bessel) for c in covs)

        mean_0 = sum(c.mean_X() for c in covs)
        mean_t = sum(c.mean_Y() for c in covs)

        for idx, c in enumerate(covs):
            if len(c.storage_XX.storage) > 0:
                c.storage_XX.storage[0].w = old_weights_xx[idx]
            if len(c.storage_XY.storage) > 0:
                c.storage_XY.storage[0].w = old_weights_xy[idx]
            if len(c.storage_YY.storage) > 0:
                c.storage_YY.storage[0].w = old_weights_yy[idx]
        return c00, c01, c11, mean_0, mean_t

    def score(self, train_covs, test_covs, k=5, scoring_method='VAMP2', return_singular_values=False):
        # split test and train test sets from input
        self.logger.debug("test set: %s\t\t train set: %s", test_covs, train_covs)
        covs_test = self.covs_[test_covs]
        covs_train = self.covs_[train_covs]
        return self._score_impl(covs_test, covs_train, scoring_method, k, return_singular_values)

    @staticmethod
    def _score_impl(test, train, scoring_method, k, return_singular_values=False):
        from pyemma.coordinates.transform.vamp import VAMPModel
        c00_test, c01_test, c11_test, mean_0_test, mean_t_test = Covariances._aggregate(test)
        c00_train, c01_train, c11_train, mean_0_train, mean_t_train = Covariances._aggregate(train)

        mean_0_diff = mean_0_test - mean_0_train
        mean_t_diff = mean_t_test - mean_t_train

        print('mean_0_diff:', mean_0_diff, np.linalg.norm(mean_0_diff))
        print('mean_t_diff:', mean_t_diff, np.linalg.norm(mean_t_diff))

        epsilon = 1e-6
        m_train = VAMPModel()
        m_train.update_model_params(dim=k, epsilon=epsilon,
                                    mean_0=mean_0_train,
                                    mean_t=mean_t_train,
                                    C00=c00_train,
                                    C0t=c01_train,
                                    Ctt=c11_train)
        m_test = VAMPModel()
        m_test.update_model_params(dim=k, epsilon=epsilon,
                                   mean_0=mean_0_test,
                                   mean_t=mean_t_test,
                                   C00=c00_test,
                                   C0t=c01_test,
                                   Ctt=c11_test)
        score = m_train.score(test_model=m_test, score_method=scoring_method)
        if return_singular_values:
            return score, m_train.singular_values, m_test.singular_values
        else:
            return score

    def score_cv(self, n=10, k=None, scoring_method='VAMP2', splitter='shuffle', return_singular_values=False,
                 n_jobs=1):

        if splitter == 'shuffle':
            splitter = _ShuffleSplit(n)
        elif not (hasattr(splitter, 'split') and callable(splitter.split)):
            raise ValueError("splitter must be either \"split\" or splitter instance with split(X) method")

        if n_jobs is None:
            from pyemma._base.parallel import get_n_jobs
            n_jobs = get_n_jobs(logger=self.logger)
        n_jobs = min(n, n_jobs)

        import psutil
        parent = psutil.Process()
        print(parent, parent.memory_full_info())
        pg = ProgressReporter()
        pg.register(n, "score cv", stage="cv")

        if n_jobs > 1:
            from multiprocess import get_context

            args = list((covs_train, covs_test, scoring_method, k, return_singular_values)
                        for covs_train, covs_test in splitter.split(self.covs_))

            def callback(score):
                pg.update(1, stage='cv')

            ctx = get_context('fork')
            # TODO: this avoids pickling the large covs_ array, but it is not very clean!
            global _covs_global
            _covs_global = self.covs_

            with ctx.Pool(n_jobs) as pool, pg.context():
                res_async = [pool.apply_async(_worker, a, callback=callback) for a in args]
                scores = [x.get() for x in res_async]
            assert scores
        else:
            scores = []
            with pg.context():
                for covs_train, covs_test in splitter.split(self.covs_):
                    scores.append(self.score(covs_train, covs_test,
                                             k=k, scoring_method=scoring_method,
                                             return_singular_values=return_singular_values))
                    pg.update(1, stage='cv')

        if return_singular_values:
            singular_values = np.array([(s[1], s[2]) for s in scores])
            scores = np.array([s[0] for s in scores])

        if return_singular_values:
            return scores, singular_values

        return scores
