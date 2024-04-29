#!/usr/bin/python3

import numpy as np
import torch

from proj2_code.student_feature_matching import match_features, compute_feature_distances


def test_compute_dists():
    """
    Test feature distance calculations.
    """
    feats1 = np.array(
        [
            [.707, .707],
            [-.707, .707], 
            [-.707, -.707]
        ])
    feats2 = np.array(
        [
            [-.5, -.866],
            [.866, -.5],
            [.5, .866],
            [-.866, .5]
        ])
    dists = np.array(
        [
            [1.98271985, 1.21742762, 0.26101724, 1.58656169],
            [1.58656169, 1.98271985, 1.21742762, 0.26101724],
            [0.26101724, 1.58656169, 1.98271985, 1.21742762]
        ])
    inter_distances = compute_feature_distances(feats1, feats2)
    assert inter_distances.shape[0] == 3
    assert inter_distances.shape[1] == 4
    assert np.allclose(dists, inter_distances, atol = 1e-03)


def test_feature_matching():
    """
    Few matches example. Match based on the following affinity/distance matrix:

        [2.  1.2 0.3 1.6]
        [1.6 2.  1.2 0.3]
        [0.3 1.6 2.  1.2]
        [1.2 0.3 1.6 2. ]
    """
    feats1 = np.array(
        [
            [.707, .707],
            [-.707, .707],
            [-.707, -.707],
            [.707, -.707]
        ])
    feats2 = np.array(
        [
            [-.5, -.866],
            [.866, -.5],
            [.5, .866],
            [-.866, .5]
        ])
    x1 = np.array([11,12,13,14])
    y1 = np.array([14,13,12,11])
    x2 = np.array([11,12,13,14])
    y2 = np.array([15,16,17,18])
    matches = np.array(
        [
            [0,2],
            [1,3],
            [2,0],
            [3,1]
        ])
    result, confidences = match_features(feats1, feats2, x1, y1, x2, y2)
    assert np.array_equal(matches, result[np.argsort(result[:, 0])])


