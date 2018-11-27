import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))


def estimate_covs(test_system, feature, lag, n_covs=30, fixed_seed=False, reader=None):
    """
    Returns
    -------
    estimated covariances instance.
    """
    if reader is None:
        import os
        print('current path: %s' % os.getcwd())
        import paths as p

        if feature == 'xyz':
            reader = p.create_cartesian_reader(test_system)
        else:
            reader = p.create_fragmented_reader(test_system, feature)

    assert np.all(reader.trajectory_lengths() > 0)
    assert reader.chunksize > 0
    from covariances import Covariances
    cov = Covariances(n_covs=n_covs, tau=lag, mode='sliding', n_save=1, fixed_seed=fixed_seed)
    cov.estimate(reader)
    return cov


def score_and_save(cov, parameters, fname, n_splits=10, k=10,
                   scoring_method='VAMP2', splitter='kfold', fixed_seed=False):
    if splitter == 'kfold':
        from sklearn.model_selection import KFold
        splitter_input = KFold(n_splits=n_splits, random_state=fixed_seed)
    elif splitter == 'shuffle':
        splitter_input = 'shuffle'
    elif splitter == 'LOO':
        from sklearn.model_selection import LeaveOneOut
        splitter_input = LeaveOneOut()
    else:
        raise ValueError('unknown splitter %s' % splitter)
    scores = np.array(cov.score_cv(n=n_splits, k=k, n_jobs=1,
                                   scoring_method=scoring_method, splitter=splitter_input,
                                   return_singular_values=False))

    print('scores:', scores.mean(), scores.std())

    np.savez_compressed(fname, parameters=parameters, scores=scores, singular_values=None)
