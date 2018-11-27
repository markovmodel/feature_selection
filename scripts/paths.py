""" This module assists in grouping the fragmented trajectories of the DESRES fast folding proteins dataset.

It also defines distance based feature transformations (contacts, exp(-d), log).


"""
import os
import re
import socket
from glob import glob

from pyemma.coordinates import featurizer, source
from pyemma.coordinates.data import FragmentedTrajectoryReader
from pyemma.coordinates.data._base.transformer import StreamingTransformer

hostname = socket.gethostname()
print('operating on host:', hostname)

# on allegro compute node
on_allegro = hostname.startswith('cmp2') or hostname =='allegro'
if on_allegro:
    RAW_TRAJS_PATH = '/home/marscher/NO_BACKUP/data/feature_sel/raw'
    PRE_CALC_FEATURES_PATH = '/home/marscher/NO_BACKUP/data/feature_sel/cached_features/'
else:
    RAW_TRAJS_PATH = '/group/ag_cmb/simulation-data/DESRES-Science2011-FastProteinFolding/'
    PRE_CALC_FEATURES_PATH = '/group/ag_cmb/marscher/feature_sel/cached_feat/'


def files_by_test_system(test_system, feature):
    from glob import glob
    files = sorted(
            glob(PRE_CALC_FEATURES_PATH + '/*{sys}*{feat}*.h5'.format(sys=test_system, feat=feature))
    )
    assert files
    return files


def create_traj_top_pairs(test_system):
    """
    groups trajectories for independent runs
    Parameters
    ----------
    test_system

    Returns
    -------
    dict {trajs: [], top: str}
    """
    # match trajectory files with topology
    if not on_allegro:
        if 'NTL9' not in test_system:
            patterns = RAW_TRAJS_PATH + '/*' + test_system + '*/*' + test_system + '*/*.dcd'
            print(patterns)
            top = glob(RAW_TRAJS_PATH + '/*' + test_system + '*/*' + test_system + '*/*.pdb')[0]
        else:
            patterns = RAW_TRAJS_PATH + '/' + test_system + '*/*.dcd'
            top = glob(RAW_TRAJS_PATH + '/' + test_system + '*/*.pdb')[0]
    else:
        if 'NTL9' not in test_system:
            patterns = RAW_TRAJS_PATH + '/*' + test_system + '*/*.dcd'
            print(patterns)
            top = glob(RAW_TRAJS_PATH + '/*' + test_system + '*/*.pdb')[0]
        else:
            patterns = RAW_TRAJS_PATH + '/' + test_system + '*/*.dcd'
            top = glob(RAW_TRAJS_PATH + '/' + test_system + '*/*.pdb')[0]

    trajs = sorted(glob(patterns))
    assert trajs

    # lambda has 4 independent runs, remove empty sub lists later on.
    files_for_features_independent_runs = [[], [], [], []]
    for f in trajs:
        m = re.match('.*{sys}-([0-9]).*'.format(sys=test_system), f)
        assert m
        ind_run = int(m.group(1))
        files_for_features_independent_runs[ind_run].append(f)
    files_for_features_independent_runs = list(filter(lambda x: x, (e for e in files_for_features_independent_runs)))

    assert top
    assert files_for_features_independent_runs
    print('top:', top)

    trajs_with_top = {'trajs': files_for_features_independent_runs, 'top': top}
    return trajs_with_top


def create_cartesian_reader(test_system):
    traj_top_pairs = create_traj_top_pairs(test_system)

    # create featurizer with alignment to first frame of dataset as reference
    def create_featurizer():
        f = featurizer(topfile=traj_top_pairs['top'])
        import mdtraj
        pattern = os.path.join(os.path.dirname(__file__), 'folded_states/*{sys}*.pdb'.format(sys=test_system.lower()))
        match = glob(pattern)
        print('aligning to:', match[0])
        assert match
        reference = mdtraj.load(match[0])
        def center(traj):
            xyz = traj.superpose(reference).xyz
            return xyz.reshape(len(xyz), -1)

        f.add_custom_func(center, dim=reference.n_atoms * 3)
        return f

    reader = source(traj_top_pairs['trajs'], features=create_featurizer())
    assert isinstance(reader, FragmentedTrajectoryReader)
    return reader


def test_create_cartesian_reader():
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
    for t in test_systems:
        create_cartesian_reader(t)


class DistTrans(StreamingTransformer):
    def __init__(self, dim, scheme='d1', c=None):
        super(DistTrans, self).__init__(None)
        self.scheme = scheme
        self.c = c
        self._dim = dim

    def _transform_array(self, X):
        import numpy as np
        if self.c is None:
            if self.scheme == 'd1':
                return 1.0 / X
            elif self.scheme == 'd2':
                return 1.0 / np.sqrt(X)
            elif self.scheme == 'expd':
                return np.exp(-X)
            elif self.scheme == 'log':
                return np.log(X)
        else:
            res = np.zeros_like(X)
            # apply cut-off
            mask = X <= self.c
            res[mask] = 1.0
            return res

    def describe(self):
        pass

    def dimension(self):
        return self._dim


def create_fragmented_reader(test_system, feature):
    if feature.startswith('res_mindist_'):
        files_for_features = files_by_test_system(test_system, 'res_mindist')
    else:
        files_for_features = files_by_test_system(test_system, feature)
    # lambda has 4 independent runs, remove empty sub lists later on.
    files_for_features_independent_runs = [[], [], [], []]

    for f in files_for_features:
        m = re.match('.*{sys}-([0-9]).*'.format(sys=test_system), f)
        assert m
        ind_run = int(m.group(1))
        files_for_features_independent_runs[ind_run].append(f)
    # remove empty sub lists.
    files_for_features_independent_runs = list(
        filter(lambda x: x, (e for e in files_for_features_independent_runs)))
    assert files_for_features_independent_runs[0]
    #  create FragmentedReader per feature, which groups all the files of a system.
    # independent runs for each test system are named like $test_system-[0-9]-feat.dcd.h5
    from pyemma.coordinates import source
    reader = source(files_for_features_independent_runs)

    # transform feature if needed (distances).
    if feature.startswith('res_mindist_'):
        name = feature
        match = re.match('res_mindist_c_(.*)', feature)
        if match is not None:
            arg = {'c': float(match.group(1))}
        else:
            arg = {'scheme': re.match('res_mindist_(.*)', name).group(1)}

        trans = DistTrans(dim=reader.ndim, **arg)
        trans.data_producer = reader
        reader = trans

    return reader


def get_output_file_name(grid, id_, include_k=True):
    params = grid[id_]
    feature = params['1_feature']
    test_system = params['0_test_system']
    lag = params['lag']
    k = params['k']
    if include_k:
        template = 'scores_test_sys_{test_system}_lag_{lag}_feat_{feature}_score_{scoring_method}_{k}.npz'
    else:
        template = 'scores_test_sys_{test_system}_lag_{lag}_feat_{feature}_score_{scoring_method}.npz'
    fname = template.format(
        test_system=test_system,
        lag=lag,
        mode='sliding',
        splitter='kfold',
        scoring_method='VAMP2',
        feature=feature,
        k=k,
    )
    return fname


def missing(grid, output_path):
    import os
    res = []
    for id_ in range(len(grid)):
        fn = get_output_file_name(grid, id_)
        path = os.path.join(output_path, fn)
        if not os.path.exists(path):
            res.append(id_)
    return res

