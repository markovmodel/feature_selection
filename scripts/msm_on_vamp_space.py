import matplotlib

matplotlib.use('Agg')

import pyemma

print('pyemma path:', pyemma.__path__)
import numpy as np
import sys
import os

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


features = ['xyz',
            'flex_torsions',
            'res_mindist_expd',
            'shrake_ruply_residue',
            'flex_torsions+best_contact',
            'flex_torsions+expd',
            'res_mindist_c_1.0',
            ]

grid = ParameterGrid([{'0_test_system': test_systems,
                       'lag': [500],  # dt=200ps, lag=100ns
                       '1_feature': features,
                       'n_centers': [None],
                       'k': [5],
                       }])

for i, p in enumerate(grid):
    print(i, p)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output')
parser.add_argument('--cov_path')
parser.add_argument('id_', type=int)
args = parser.parse_args()

output_path = args.output

if not os.path.exists(output_path):
    sys.exit(23)


def get_output_file_name(grid, id_):
    params = grid[id_]
    feature = params['1_feature']
    test_system = params['0_test_system']
    n_states = params['n_centers']

    fname_its = os.path.join(output_path, 'its_test_sys_{test_system}_nstates_{n_states}_feat_{feature}.pdf'.format(
        test_system=test_system,
        feature=feature,
        n_states=n_states
    ))
    fname_cktest = os.path.join(output_path,
                                'cktest_test_sys_{test_system}_nstates_{n_states}_feat_{feature}.pdf'.format(
                                    test_system=test_system,
                                    feature=feature,
                                    n_states=n_states
                                ))
    fname_cluster = os.path.join(output_path, 'cl_test_sys_{sys}_nstates_{n_states}_feat_{feat}.h5'.format(
        sys=test_system, n_states=n_states, feat=feature))

    fname_msm = os.path.join(output_path,
                             'msm_score_test_sys_{sys}_lag_{lag}_nstates_{n_states}_feat_{feat}.npz'.format(
                                 sys=test_system, lag=params['lag'], feat=feature, n_states=n_states,
                             ))

    from collections import namedtuple
    t = namedtuple("output_names", ["its", "ck", "cl", "msm"])
    return t(fname_its, fname_cktest, fname_cluster, fname_msm)


def _recreate_vamp_obj_from_covs(grid, id_):
    lag = grid[id_]['lag']
    v = pyemma.coordinates.vamp(lag=lag, dim=5)
    from paths import get_output_file_name as out_name_score
    fname_cov = os.path.join(args.cov_path, out_name_score(grid, id_, include_k=False) + '_covs.h5')
    covs = pyemma.load(fname_cov)
    c00, c01, c11, mean_0, mean_t = covs._aggregate(covs.covs_)
    v.model.C00 = c00
    v.model.C0t = c01
    v.model.Ctt = c11
    v.model.mean_0 = mean_0
    v.model.mean_t = mean_t
    v._estimated = True
    return v


def run(id_):
    print('job number: ', id_)
    params = grid[id_]
    test_system = params['0_test_system']
    feature = grid[id_]['1_feature']
    lag = params['lag']

    print('current path: %s' % os.getcwd())
    try:
        import paths as p
        if feature == 'xyz':
            reader = p.create_cartesian_reader(test_system)
        elif feature == 'flex_torsions+expd':
            from calc_cov_expd_torsions import get_expd_flex_torsions_reader
            reader = get_expd_flex_torsions_reader(test_system)
        elif feature.find('_c') != 0:
            from calc_cov_combined_feat import best_scoring_contact_feature
            from calc_cov_combined_feat import get_best_contact_flex_torsions_reader
            best_scoring_contact_file, c = best_scoring_contact_feature(test_system=test_system, score='VAMP2', k=5,
                                                                        output_path=args.cov_path)
            reader = get_best_contact_flex_torsions_reader(c, test_system)
        else:
            reader = p.create_fragmented_reader(test_system, feature)

        assert np.all(reader.trajectory_lengths() > 0)
        assert reader.chunksize > 0

        t = _recreate_vamp_obj_from_covs(grid, id_=id_)
        t.data_producer = reader
        one_frame = next(t.iterator(chunk=1, return_trajindex=False))
        assert one_frame.shape[1] == 5
        y = t.get_output()

        for real_k in (50, 250,  500, 1000, 2000, 3000):
            print('process clustering for k=%s' % real_k)
            grid.param_grid[0]['n_centers'] = [real_k]
            out_names = get_output_file_name(grid=grid, id_=id_)

            fname_its = out_names.its
            if os.path.exists(fname_its):
                print("results file %s already exists. Skipping" % fname_its)

            fname_cluster = out_names.cl
            if os.path.exists(fname_cluster):
                print('loading clustering from', fname_cluster)
                cluster = pyemma.load(fname_cluster)
                if len(cluster._dtrajs) == 0:
                    dtrajs = cluster.assign(y)
                    cluster._dtrajs = dtrajs
                    cluster.save(fname_cluster, overwrite=True)
            else:
                cluster = pyemma.coordinates.cluster_kmeans(data=y, k=real_k, max_iter=100, stride=1, chunksize=0,
                                                            n_jobs=16)

                dtrajs = cluster.assign(y)
                cluster._dtrajs = dtrajs
                cluster.save(fname_cluster)

            msm = pyemma.msm.estimate_markov_model(cluster.dtrajs, lag=lag)
            scores_msm = msm.score_cv(cluster.dtrajs, score_k=5, n=50)
            print('msm scores n_centers(%s):' % real_k, scores_msm, scores_msm.mean(), scores_msm.std())
            np.save(out_names.msm, arr=scores_msm)

    except BaseException as e:
        print('bad:', e, id_)
        import traceback
        traceback.print_exc(file=sys.stdout)
        import pdb
        pdb.post_mortem()
        raise
    else:
        print('successful')


if __name__ == '__main__':
    run(args.id_)
