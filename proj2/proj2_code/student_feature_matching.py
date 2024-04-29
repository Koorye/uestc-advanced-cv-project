import numpy as np


def compute_feature_distances(features1, features2):
    """
    This function computes a list of distances from every feature in one array
    to every feature in another.
    Args:
    - features1: A numpy array of shape (n,feat_dim) representing one set of
      features, where feat_dim denotes the feature dimensionality
    - features2: A numpy array of shape (m,feat_dim) representing a second set
      features (m not necessarily equal to n)

    Returns:
    - dists: A numpy array of shape (n,m) which holds the distances from each
      feature in features1 to each feature in features2
    """
    mode = 'euclidean'

    # Euclidean distance
    # D = \sqrt{\sum_{i=1}^{D} (x_i - y_i)^2}
    # (N, 1, D) - (1, M, D) -> (N, M, D) -> (N, M)
    if mode == 'euclidean':
        return np.sqrt(np.sum((features1[:, None] - features2) ** 2, axis=2))

    # Manhattan distance
    # D = \sum_{i=1}^{D} |x_i - y_i|
    if mode == 'manhattan':
        return np.sum(np.abs(features1[:, None] - features2), axis=2)

    # Chebyshev distance
    # D = \max_{i=1}^{D} |x_i - y_i|
    if mode == 'chebyshev':
        return np.max(np.abs(features1[:, None] - features2), axis=2)
    
    # Minowski distance
    # D = \left( \sum_{i=1}^{D} |x_i - y_i|^p \right)^{1/p}
    if mode == 'minowski':
        p = 3
        return np.sum(np.abs(features1[:, None] - features2) ** p, axis=2) ** (1 / p)


def match_features(features1, features2, x1, y1, x2, y2):
    """
    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).

    To start with, simply implement the "ratio test", equation 4.18 in
    section 4.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper).

    You should call `compute_feature_distances()` in this function, and then
    process the output.

    Args:
    - features1: A numpy array of shape (n,feat_dim) representing one set of
      features, where feat_dim denotes the feature dimensionality
    - features2: A numpy array of shape (m,feat_dim) representing a second
      set of features (m not necessarily equal to n)
    - x1: A numpy array of shape (n,) containing the x-locations of features1
    - y1: A numpy array of shape (n,) containing the y-locations of features1
    - x2: A numpy array of shape (m,) containing the x-locations of features2
    - y2: A numpy array of shape (m,) containing the y-locations of features2

    Returns:
    - matches: A numpy array of shape (k,2), where k is the number of matches.
      The first column is an index in features1, and the second column is an
      index in features2
    - confidences: A numpy array of shape (k,) with the real valued confidence
      for every match

    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """
    
    def mean_distance_filter(dists, alpha):
        """
        Generate mask for matching features based on mean distance.
        Params:
        - dists (M,)
        Return:
        - mask (M,)
        """
        mean_dist = np.mean(dists)
        std_dist = np.std(dists)
        return dists < mean_dist - alpha * std_dist
    
    
    def mean_cosine_filter(cosines, alpha):
        """
        Generate mask for matching features based on mean cosine similarity.
        Params:
        - cosines (M,)
        Return:
        - mask (M,)
        """
        mean_cosine = np.mean(cosines)
        std_cosine = np.std(cosines)
        return cosines > mean_cosine + alpha * std_cosine
    
    
    def dist_ratio_filter(best_match_dists, second_match_dists, thresh):
        """
        Generate mask for matching features based on distance ratio.
        Params:
        - best_match_dists (M,)
        - second_match_dists (M,)
        Return:
        - mask (M,)
        """
        return (best_match_dists / second_match_dists) < thresh
    
    
    def cross_validation_filter(best_matches, inv_best_matches):
        """
        Generate mask for matching features based on cross validation.
        Params:
        - best_matches (M,): best matches for features1, elements are indices of features2
        - inv_best_matches (N,): best matches for features2, elements are indices of features1
        Return:
        - mask (M,)
        """
        return inv_best_matches[best_matches] == np.arange(best_matches.shape[0])
    
    
    def spatial_filter(shifts, cosine_thresh, dist_thresh):
        """
        Generate mask for matching features based on shifts.
        Params:
        - shifts (M, 2): shifts of features1 to features2
        Return:
        - mask (M,)
        """
        # cosine mask
        # (1, 2)
        mean_direction = np.mean(shifts, axis=0, keepdims=True)
        # (M, 2), (1, 2) -> (M, 2)
        cosine = np.sum(shifts * mean_direction, axis=1) / (np.linalg.norm(shifts, axis=1) * np.linalg.norm(mean_direction, axis=1))
        cosine_mask = cosine > cosine_thresh

        # distance mask
        # (M,)
        dists = np.linalg.norm(shifts, axis=1)
        mean_dist = np.mean(dists)
        std_dist = np.std(dists)
        # between mean - std and mean + std
        dist_mask = (dists > mean_dist - dist_thresh * std_dist) & (dists < mean_dist + dist_thresh * std_dist)

        # return cosine_mask
        return cosine_mask & dist_mask
    
    
    def ranasc_filter():
        """
        Generate mask for matching features based on RANSAC.
        """
        pass

    mode = 'euclidean'
    dist_alpha = 0.1
    cosine_alpha = 0.7
    ratio_thresh = 0.8
    spatial_cosine_thresh = 0.6
    spatial_dist_thresh = 1.0
    
    print('Computing distances...')
    # (M, N)
    dists = compute_feature_distances(features1, features2)

    cosines = np.dot(features1, features2.T) / (np.linalg.norm(features1) * np.linalg.norm(features2))
    
    best_matches = np.argmin(dists, axis=1)
    inv_best_matches = np.argmin(dists, axis=0)

    best_match_dists = dists[np.arange(dists.shape[0]), best_matches]
    second_match_dists = dists[np.arange(dists.shape[0]), np.argsort(dists, axis=1)[:, 1]]
    best_match_cosines = cosines[np.arange(cosines.shape[0]), best_matches]

    if best_match_dists.shape[0] < 100:
        inds1, inds2 = np.arange(features1.shape[0]), best_matches
        matches = np.stack([inds1, inds2], axis=1)
        confidences = 1 / best_match_dists
        return matches, confidences
    
    dist_mask = mean_distance_filter(best_match_dists, dist_alpha)
    print('Distance filter dropped:', np.sum(~dist_mask))

    cosine_mask = mean_cosine_filter(best_match_cosines, cosine_alpha)
    print('Cosine filter dropped:', np.sum(~cosine_mask))

    ratio_mask = dist_ratio_filter(best_match_dists, second_match_dists, ratio_thresh)
    print('Ratio filter dropped:', np.sum(~ratio_mask))

    cross_val_mask = cross_validation_filter(best_matches, inv_best_matches)
    print('Cross validation filter dropped:', np.sum(~cross_val_mask))

    mask = dist_mask & cosine_mask & ratio_mask & cross_val_mask
    inds1, inds2 = np.arange(features1.shape[0]), best_matches
    matches = np.stack([inds1[mask], inds2[mask]], axis=1)
    confidences = 1 / best_match_dists[mask]

    # (M,) -> (M',) 
    x1, y1 = x1[matches[:, 0]], y1[matches[:, 0]]
    x2, y2 = x2[matches[:, 1]], y2[matches[:, 1]]
    shifts_x = x1 - x2
    shifts_y = y1 - y2
    shifts = np.stack([shifts_x, shifts_y], axis=1)

    # spatial_mask = spatial_filter(shifts, spatial_cosine_thresh, spatial_dist_thresh)
    # print('Spatial filter dropped:', np.sum(~spatial_mask))

    # matches = matches[spatial_mask]
    # confidences = confidences[spatial_mask]
    
    return matches, confidences
