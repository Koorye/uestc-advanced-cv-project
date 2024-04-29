from proj4_code.stats_helper import compute_mean_and_std

import numpy as np

import os


def test_mean_and_variance():
  if os.path.exists('../data/'):
  	mean, std = compute_mean_and_std('../data/')
  else:
  	mean, std = compute_mean_and_std('data/')
  	

  assert np.allclose(mean, np.array([0.45547487]))
  assert np.allclose(std, np.array([0.25316328]))
