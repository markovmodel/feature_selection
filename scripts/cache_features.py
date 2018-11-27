""" This module defines functions to pre-compute the features from the raw trajectories

The extracted features have the name pattern {traj}_{feat}.h5, where traj corresponds
to a fragment of the DESRES fast folding proteins data set.

The files are in HDF5 format with a BLOSC compression, which requires the BLOSC plugin to be loaded.
The latter is being done by importing tables prior h5py.

"""

import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


from paths import create_traj_top_pairs

import tables # for BLOSC support.
import h5py

import numpy as np
import pyemma

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
                'NTL9'
]

#### features
def shrake_ruply(featurizer):
    def featurize(traj):
        import mdtraj
        res = mdtraj.shrake_rupley(traj, probe_radius=0.14, n_sphere_points=960, mode='residue')
        return res

    featurizer.add_custom_func(featurize, dim=featurizer.topology.n_atoms)

def backbone(featurizer):
    featurizer.add_backbone_torsions(cossin=True)

def dist_ca(featurizer):
    featurizer.add_distances_ca()

def res_mindist(featurizer):
    featurizer.add_residue_mindist(scheme='closest-heavy')

def side_sidechain_torsions(featurizer):
    print("adding feature sidechain torssions to featurizer")
    from mdtraj.geometry.dihedral import indices_chi1, indices_chi2, indices_chi3, indices_chi4, indices_chi5, \
        indices_omega
    top = featurizer.topology
    indices = np.vstack((indices_chi1(top),
                         indices_chi2(top),
                         indices_chi3(top),
                         indices_chi4(top),
                         indices_chi5(top),
                         indices_omega(top)))
    assert indices.shape[1] == 4
    from mdtraj import compute_dihedrals

    def compute_side_chains(traj):
        res = compute_dihedrals(traj, indices)
        # cossin
        rad = np.dstack((np.cos(res), np.sin(res)))
        rad = rad.reshape(rad.shape[0], rad.shape[1] * rad.shape[2])
        print('shape chunk:',rad.shape)
        return rad

    featurizer.add_custom_func(compute_side_chains, dim=len(indices)*2)


def dssp(featurizer):
    dim = featurizer.topology.n_residues
    from mdtraj import compute_dssp
    codes = ('H', 'B', 'E', 'G', 'I', 'T', 'S', ' ', 'NA')
    #codes_simple = ('H', 'E', 'C', 'NA')

    mapping = {}
    for i, c in enumerate(codes):
        vec = np.zeros_like(codes, dtype=np.float32)
        vec[i] = 1
        mapping[c] = vec
    #
    #mapping['NA'] = np.zeros_like(codes, dtype=np.float32)

    def _compute(traj):
        result = np.empty((len(traj), traj.topology.n_residues, len(codes)), dtype=np.float32)
        result[:] = np.nan
        assignment = compute_dssp(traj, simplified=False)
        for i, a_ in enumerate(assignment):
            for j, a in enumerate(a_):
                val = mapping.get(a, None)
                if val is None:
                    raise Exception('code at position %s could not be decoded: "%s"' % (i,c))
                result[i, j, :] = val

        result = result.reshape((len(traj), -1))
        assert np.all(np.isfinite(result))
        return result

    featurizer.add_custom_func(_compute, dim=dim*len(codes))


features = (
    side_sidechain_torsions,
    res_mindist,
    backbone,
    shrake_ruply,
)


from sklearn.model_selection import ParameterGrid
import itertools
param_grid = {'test_sys': test_systems,
              'feature': features,
              }

grid = ParameterGrid(param_grid)
# this should be equal to njobs for slurm!
print("num total jobs:", len(list(grid)))

def run(id):
    print('job number: %s' % id)

    try:
        import os
        os.chdir('/home/marscher/NO_BACKUP/data/feature_sel/cached_features/')
        params = grid[id]
        trajs_top = create_traj_top_pairs(test_system=params['test_sys'])
        trajs_flat = list(itertools.chain(*trajs_top['trajs']))
        top = trajs_top['top']
        feature = params['feature']
        feature_name = feature.__name__

        for traj in trajs_flat:
            file_name = "{traj}_{feat}.h5".format(traj=os.path.basename(traj), feat=feature_name)
            print('top: {} \n\ntrajs: {}'.format(top, traj))
            if os.path.exists(file_name):
                print('exists')
                continue
            reader = pyemma.coordinates.source(traj, top=top)
            featurizer = reader.featurizer
            # add feature to featurizer.
            feature(featurizer)

            print("writing to:", file_name)
            # TODO: this is now implemented in PyEMMA (write_to_hdf5) and could be used here instead.
            with h5py.File(name=file_name, mode='a') as f:
                try:
                    dataset = f.create_dataset(feature_name, shape=(reader.n_frames_total(), reader.ndim),
                                               compression=32001, chunks=True, shuffle=True)
                    dataset[:] = np.nan
                except RuntimeError:
                    print("data set exists already, skipping")
                    return

                t = 0
                with reader.iterator(return_trajindex=False) as it:
                    for chunk in it:
                        dataset[t:t+len(chunk)] = chunk
                        t += len(chunk)

                assert np.all(np.isfinite(dataset[:]))

    except BaseException as e:
        print('bad:', e, id)
        import pdb
        pdb.post_mortem()

    print('successful')


if __name__ == '__main__':
    id = int(sys.argv[1])
    run(id)
