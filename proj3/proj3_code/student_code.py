import numpy as np
import torch
import cv2
from proj3_code.feature_matching.SIFTNet import get_siftnet_features


def pairwise_distances(X, Y):
    """
    This method will be very similar to the pairwise_distances() function found
    in sklearn
    (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html)
    However, you are NOT allowed to use any library functions like this
    pairwise_distances or pdist from scipy to do the calculation!

    The purpose of this method is to calculate pairwise distances between two
    sets of vectors. The distance metric we will be using is 'euclidean',
    which is the square root of the sum of squares between every value.
    (https://en.wikipedia.org/wiki/Euclidean_distance)

    Useful functions:
    -   np.linalg.norm()

    Args:
    -   X: N x d numpy array of d-dimensional features arranged along N rows
    -   Y: M x d numpy array of d-dimensional features arranged along M rows

    Returns:
    -   D: N x M numpy array where d(i, j) is the distance between row i of X and
        row j of Y
    """
    return np.linalg.norm(X[:, np.newaxis, :] - Y[np.newaxis, :, :], axis=2)

def get_tiny_images(image_arrays):
    """
    This feature is inspired by the simple tiny images used as features in
    80 million tiny images: a large dataset for non-parametric object and
    scene recognition. A. Torralba, R. Fergus, W. T. Freeman. IEEE
    Transactions on Pattern Analysis and Machine Intelligence, vol.30(11),
    pp. 1958-1970, 2008. http://groups.csail.mit.edu/vision/TinyImages/

    To build a tiny image feature, simply resize the original image to a very
    small square resolution, e.g. 16x16. You can either resize the images to
    square while ignoring their aspect ratio or you can crop the center
    square portion out of each image. Making the tiny images zero mean and
    unit length (normalizing them) will increase performance modestly.

    Useful functions:
    -   cv2.resize
    -   ndarray.flatten()

    Args:
    -   image_arrays: list of N elements containing image in Numpy array, in
                grayscale

    Returns:
    -   feats: N x d numpy array of resized and then vectorized tiny images
                e.g. if the images are resized to 16x16, d would be 256
    """
    def resize(img, size, keep_aspect_ratio):
        if not keep_aspect_ratio:
            img = cv2.resize(img, (size, size))
        else:
            h, w = img.shape
            if h > w:
                img = cv2.resize(img, (size, int(size * h / w)))
                img = img[(img.shape[0] - size) // 2:(img.shape[0] + size) // 2, :]
            else:
                img = cv2.resize(img, (int(size * w / h), size))
                img = img[:, (img.shape[1] - size) // 2:(img.shape[1] + size) // 2]
        return img
    
    def normalize(img):
        img = img.flatten()
        img = img - np.mean(img)
        img = img / np.linalg.norm(img)
        return img
    
    feats = []
    keep_aspect_ratio = False
    size = 16

    for img in image_arrays:
        resized_img = resize(img, size, keep_aspect_ratio)
        feats.append(normalize(resized_img))

    return np.array(feats)


def nearest_neighbor_classify(train_image_feats, train_labels,
                              test_image_feats, k=3):
    """
    This function will predict the category for every test image by finding
    the training image with most similar features. Instead of 1 nearest
    neighbor, you can vote based on k nearest neighbors which will increase
    performance (although you need to pick a reasonable value for k).
    Useful functions:
    -   D = pairwise_distances(X, Y)
          computes the distance matrix D between all pairs of rows in X and Y.
            -  X is a N x d numpy array of d-dimensional features arranged along
            N rows
            -  Y is a M x d numpy array of d-dimensional features arranged along
            N rows
            -  D is a N x M numpy array where d(i, j) is the distance between row
            i of X and row j of Y
    Args:
    -   train_image_feats:  N x d numpy array, where d is the dimensionality of
            the feature representation
    -   train_labels: N element list, where each entry is a string indicating
            the ground truth category for each training image
    -   test_image_feats: M x d numpy array, where d is the dimensionality of the
            feature representation. You can assume N = M, unless you have changed
            the starter code
    -   k: the k value in kNN, indicating how many votes we need to check for
            the label
    Returns:
    -   test_labels: M element list, where each entry is a string indicating the
            predicted category for each testing image
    """
    
    test_labels = []
    
    # (M, N) -> (M, K)
    dists = pairwise_distances(test_image_feats, train_image_feats)
    topk_inds = np.argsort(dists, axis=1)[:, :k]
    topk_labels = np.array(train_labels)[topk_inds]

    for labels in topk_labels:
        label, count = np.unique(labels, return_counts=True)
        test_labels.append(label[np.argmax(count)])
    
    return test_labels


def kmeans(feature_vectors, k, max_iter = 10):
    """
    Implement the k-means algorithm in this function. Initialize your centroids
    with random *unique* points from the input data, and repeat over the
    following process:
    1. calculate the distances from data points to the centroids
    2. assign them labels based on the distance - these are the clusters
    3. re-compute the centroids from the labeled clusters

    Please note that you are NOT allowed to use any library functions like
    vq.kmeans from scipy or kmeans from vlfeat to do the computation!

    Useful functions:
    -   np.random.randint
    -   np.linalg.norm
    -   np.argmin

    Args:
    -   feature_vectors: the input data collection, a Numpy array of shape (N, d)
            where N is the number of features and d is the dimensionality of the
            features
    -   k: the number of centroids to generate, of type int
    -   max_iter: the total number of iterations for k-means to run, of type int

    Returns:
    -   centroids: the generated centroids for the input feature_vectors, a Numpy
            array of shape (k, d)
    """
    N, D = feature_vectors.shape
    labels_prev = np.zeros(N)
    
    # random initialization
    # inds = np.random.choice(N, k, replace=False)

    # initialization with KMeans++
    inds = [np.random.choice(N)]
    for _ in range(k - 1):
        dists = np.min(pairwise_distances(feature_vectors, feature_vectors[inds]), axis=1)
        probs = dists / np.sum(dists)
        inds.append(np.random.choice(N, p=probs))

    # (K, D)
    centroids = feature_vectors[inds]

    for _ in range(max_iter):
        # (N, K) -> (N, )
        # assign labels
        dists = pairwise_distances(feature_vectors, centroids)
        labels = np.argmin(dists, axis=1)
        
        # check convergence
        if np.all(labels == labels_prev):
            break
        labels_prev = labels
        
        # update centroids
        for i in range(k):
            if np.sum(labels == i) == 0:
                continue
            centroids[i] = np.mean(feature_vectors[labels == i], axis=0)
    
    return centroids


def build_vocabulary(image_arrays, vocab_size, stride = 20):
    """
    This function will sample SIFT descriptors from the training images,
    cluster them with kmeans, and then return the cluster centers.

    Load images from the training set. To save computation time, you don't
    necessarily need to sample from all images, although it would be better
    to do so. You can randomly sample the descriptors from each image to save
    memory and speed up the clustering. For testing, you may experiment with
    larger stride so you just compute fewer points and check the result quickly.

    In order to pass the unit test, leave out a 10-pixel margin in the image,
    that is, start your x and y from 10, and stop at len(image_width) - 10 and
    len(image_height) - 10.

    For each loaded image, get some SIFT features. You don't have to get as
    many SIFT features as you will in get_bags_of_sifts, because you're only
    trying to get a representative sample here.

    Once you have tens of thousands of SIFT features from many training
    images, cluster them with kmeans. The resulting centroids are now your
    visual word vocabulary.

    Note that the default vocab_size of 50 is sufficient for you to get a decent
    accuracy (>40%), but you are free to experiment with other values.

    Useful functions:
    -   np.array(img, dtype='float32'), torch.from_numpy(img_array), and
            img_tensor = img_tensor.reshape(
                (1, 1, img_array.shape[0], img_array.shape[1]))
            for converting a numpy array to a torch tensor for siftnet
    -   get_siftnet_features() from SIFTNet: you can pass in the image tensor in
            grayscale, together with the sampled x and y positions to obtain the
            SIFT features
    -   np.arange() and np.meshgrid(): for you to generate the sample x and y
            positions faster

    Args:
    -   image_arrays: list of images in Numpy arrays, in grayscale
    -   vocab_size: size of vocabulary
    -   stride: the stride of your SIFT sampling

    Returns:
    -   vocab: This is a (vocab_size, dim) Numpy array (vocabulary). Where dim
            is the length of your SIFT descriptor. Each row is a cluster center
            / visual word.
    """
    pad = 10
    keep_ratio = 1.0
    from tqdm import tqdm
    
    # sample SIFT descriptors
    sifts = []
    for img in tqdm(image_arrays, desc='build_vocabulary'):
        h, w = img.shape
        xx, yy = np.meshgrid(np.arange(pad, w - pad, stride), np.arange(pad, h - pad, stride))
        xx, yy = xx.flatten(), yy.flatten()
        # random sampling
        if 0.0 < keep_ratio < 1.0:
            inds = np.random.choice(len(xx), int(len(xx) * keep_ratio), replace=False)
            xx, yy = xx[inds], yy[inds]
        
        img_tensor = torch.from_numpy(np.array(img, dtype='float32'))
        img_tensor = img_tensor.reshape(1, 1, img.shape[0], img.shape[1])
        siftnet_feats = get_siftnet_features(img_tensor, xx, yy)
        sifts.append(siftnet_feats)
                
    # k-means clustering
    sifts = np.concatenate(sifts)
    vocab = kmeans(sifts, vocab_size)
    return vocab


def kmeans_quantize(raw_data_pts, centroids):
    """
    Implement the k-means quantization in this function. Given the input data
    and the centroids, assign each of the data entry to the closest centroid.

    Useful functions:
    -   pairwise_distances
    -   np.argmin

    Args:
    -   raw_data_pts: the input data collection, a Numpy array of shape (N, d)
            where N is the number of input data, and d is the dimension of it,
            given the standard SIFT descriptor, d = 128
    -   centroids: the generated centroids for the input feature_vectors, a
            Numpy array of shape (k, D)

    Returns:
    -   indices: the index of the centroid which is closest to the data points,
            a Numpy array of shape (N, )

    """
    # (N, K) -> (N, )
    dists = pairwise_distances(raw_data_pts, centroids)
    indices = np.argmin(dists, axis=1)
    return indices


def get_bags_of_sifts(image_arrays, vocabulary, step_size = 10):
    """
    This feature representation is described in the lecture materials,
    and Szeliski chapter 14.
    You will want to construct SIFT features here in the same way you
    did in build_vocabulary() (except for possibly changing the sampling
    rate) and then assign each local feature to its nearest cluster center
    and build a histogram indicating how many times each cluster was used.
    Don't forget to normalize the histogram, or else a larger image with more
    SIFT features will look very different from a smaller version of the same
    image.

    Useful functions:
    -  np.array(img, dtype='float32'), torch.from_numpy(img_array), and
            img_tensor = img_tensor.reshape(
                (1, 1, img_array.shape[0], img_array.shape[1]))
            for converting a numpy array to a torch tensor for siftnet
    -   get_siftnet_features() from SIFTNet: you can pass in the image tensor
            in grayscale, together with the sampled x and y positions to obtain
            the SIFT features
    -   np.histogram() or np.bincount(): easy way to help you calculate for a
            particular image, how is the visual words span across the vocab


    Args:
    -   image_arrays: A list of input images in Numpy array, in grayscale
    -   vocabulary: A numpy array of dimensions:
            vocab_size x 128 where each row is a kmeans centroid
            or visual word.
    -   step_size: same functionality as the stride in build_vocabulary(). Feel
            free to experiment with different values, but the rationale is that
            you may want to set it smaller than stride in build_vocabulary()
            such that you collect more features from the image.

    Returns:
    -   image_feats: N x d matrix, where d is the dimensionality of the
            feature representation. In this case, d will be equal to the number
            of clusters or equivalently the number of entries in each image's
            histogram (vocab_size) below.
    """
    feats = []
    pad = 10
    
    from tqdm import tqdm
    for img in tqdm(image_arrays, desc='get_bags_of_sifts'):
        # sample SIFT descriptors
        h, w = img.shape
        xx, yy = np.meshgrid(np.arange(pad, w - pad, step_size), np.arange(pad, h - pad, step_size))
        xx, yy = xx.flatten(), yy.flatten()
        
        img_tensor = torch.from_numpy(np.array(img, dtype='float32'))
        img_tensor = img_tensor.reshape((1, 1, img.shape[0], img.shape[1]))
        siftnet_feats = get_siftnet_features(img_tensor, xx, yy)

        # quantize SIFT descriptors
        indices = kmeans_quantize(siftnet_feats, vocabulary)
        hist, _ = np.histogram(indices, bins=np.arange(len(vocabulary) + 1))
        hist = hist / np.linalg.norm(hist)
        feats.append(hist)
    
    return np.array(feats)
