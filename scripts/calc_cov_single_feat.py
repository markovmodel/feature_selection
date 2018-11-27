""" This module computes the VAMP projection and scores for multiple features and test systems.

First the fragments are being grouped to match one time realisation, e.g. in lambda we have four
fragmented trajectories and passed on to a reader. In case of xyz, we align to a reference structure.
For the other features, we group the pre-computed fragments of cached features (see paths.py).


The reader is then used to estimate the Covariances object, which also does the scoring (see estimate.py).

The scores are written to a results file of the following pattern:

'scores_test_sys_{test_system}_lag_{lag}_feat_{feature}_score_{scoring_method}.npz'

"""

import os
import sys

import numpy as np
import pyemma

print('pyemma path:', pyemma.__path__)

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from pyemma._ext.sklearn.parameter_search import ParameterGrid

test_systems = [
   '1FME',
   '2F4K',
   '2JOF',
   '2WAV',
   'A3D',
   'CLN025',
   'GTT',
   'lambda',
   'NuG2',
   'PRB',
   'UVF',
   'NTL9',
]

features = (
    'xyz',
    'flex_torsions',
    'shrake_ruply',
    'res_mindist',
    'res_mindist_d1',
    'res_mindist_d2',
    'res_mindist_expd',
    'res_mindist_log',
    'res_mindist_c_0.4',
    'res_mindist_c_0.5',
    'res_mindist_c_0.6',
    'res_mindist_c_0.8',
    'res_mindist_c_1.0',
)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--output', default='/group/ag_cmb/marscher/feature_sel/')
parser.add_argument('--lag', type=int, default=500) # dt=200ps, lag=500ns
parser.add_argument('id_', type=int)
args = parser.parse_args()

output_path = args.output
assert os.path.exists(output_path)

# prefix test sys and feature to force sorting group these.
grid = ParameterGrid([{'0_test_system': test_systems,
                       'lag': [args.lag],  # dt=200ps, lag=500ns
                       '1_feature': list(features),
                       'k': [5],
                       }])


def run(id_):
    print('job number: ', id_)
    params = grid[id_]
    feature = params['1_feature']
    test_system = params['0_test_system']
    lag = params['lag']
    import estimate as e
    from paths import get_output_file_name
    cov = None
    k = params['k']

    fname = get_output_file_name(grid, id_)
    fname = os.path.join(output_path, fname)
    # exclude k param from cov file name to avoid recomp.
    fname_cov = os.path.join(output_path, get_output_file_name(grid, id_, include_k=False) + '_covs.h5')
    print('output file: %s' % fname)
    if os.path.exists(fname):
        print("results file %s already exists. Skipping" % fname)
        return
    parameters = np.array({'lag': lag,
                           'splitter': 'kfold',
                           'mode': 'sliding',
                           'test_system': test_system,
                           'scoring_method': 'vamp2',
                           'feature': feature,
                           'k': k
                           })

    if not os.path.exists(fname_cov):
        print('estimating covs')
        cov = e.estimate_covs(test_system=test_system, feature=feature, lag=lag, n_covs=50)
        cov.save(fname_cov)
    elif cov is None:
        print('loading covs from', fname_cov)
        cov = pyemma.load(fname_cov)
    assert cov is not None
    e.score_and_save(cov,
                     splitter='kfold',
                     fname=fname,
                     k=k,
                     fixed_seed=False,
                     scoring_method='VAMP2',
                     parameters=parameters, n_splits=50)


if __name__ == '__main__':
    run(args.id_)
