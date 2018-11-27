These scripts perform the computation and caching of the features,
compute/caching covariances and cross-validated VAMP scoring.

First one needs to obtain the DESRES fast folders data set, which is
copyrighted and needed to be licensed first (it is free of charge for
academic use).

After the data set has been extracted, one should point the path
definitions in paths.py.

Then one wants to invoke cache_features.py to compute
and cache the features.

After this step we invoke the calc_cov_single_feat.py script, which computes
the VAMP scores for not combined features.
The calc_cov_combined_features.py script computes the latter for the combined
features of flexible torsions and exp(-d) transformation or best scoring
contact feature.

Finally in msm_on_vamp_space, we run a MSM analysis. we project onto
the VAMP basis computed earlier and cluster via kmeans for different
centers. This discretization is then evaluated by a VAMP score
computed on MSM.

