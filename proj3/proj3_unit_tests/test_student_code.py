#!/usr/bin/python3
import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import math
from proj3_code import student_code


def verify(function) -> str:
    """ Will indicate with a print statement whether assertions passed or failed
      within function argument call.

      Args:
      - function: Python function object

      Returns:
      - string
    """
    try:
        function()
        return "\x1b[32m\"Correct\"\x1b[0m"
    except AssertionError:
        return "\x1b[31m\"Wrong\"\x1b[0m"


def test_pairwise_distances():
    """
    Testing pairwise_distances()
    """
    # Test case 1
    actual_distances = np.array([[0, math.sqrt(2), 2]])

    X = np.array([[1, 1, 1, 1]])
    Y = np.array([[1, 1, 1, 1],
                  [1, 2, 1, 2],
                  [2, 2, 2, 2]])

    test_distances = student_code.pairwise_distances(X, Y)

    # Test case 2
    actual_distances_1 = np.array([[math.sqrt(2), 1., math.sqrt(5)],
                                   [math.sqrt(3), 2., math.sqrt(2)],
                                   [math.sqrt(2), math.sqrt(5), math.sqrt(3)]])
    X_1 = np.array([[1, 1, 2],
                    [2, 1, 0],
                    [0, 1, 1]])
    Y_1 = np.array([[1, 2, 1],
                    [2, 1, 2],
                    [1, 0, 0]])

    test_distances_1 = student_code.pairwise_distances(X_1, Y_1)

    assert np.array_equal(actual_distances, test_distances)
    assert np.array_equal(actual_distances_1, test_distances_1)


def test_get_tiny_images_size():
    im1 = np.ones((480, 640))
    im2 = np.ones((1920, 1080))

    images = [im1, im2]
    tiny_images = student_code.get_tiny_images(images)
    assert tiny_images.shape == (2, 256)


def test_get_tiny_images_values():
    image = np.zeros((40, 100))
    image[:20, :50] += 1
    image[:20, 50:] += 2
    image[20:, :50] += 3
    image[20:, 50:] += 4
    images = [image]

    tiny_images = student_code.get_tiny_images(images)
    # print(tiny_images[:, :64])
    assert tiny_images[0, 0] == np.min(tiny_images)
    assert tiny_images[0, 119] == np.min(tiny_images)
    assert tiny_images[0, 120] != np.min(tiny_images)
    assert tiny_images[0, -1] == np.max(tiny_images)
    assert tiny_images[0, -120] == np.max(tiny_images)
    assert tiny_images[0, -121] != np.max(tiny_images)


def test_nearest_neighbor_classify():
    training_data = np.ones((30, 128))
    training_data[:10, :] = np.random.randint(25, 75, (10, 128))
    training_data[10:20, :] = np.random.randint(225, 275, (10, 128))
    training_data[20:30, :] = np.random.randint(425, 475, (10, 128))

    testing_data = np.ones((3, 128))
    testing_data[0, :] *= 50
    testing_data[1, :] *= 250
    testing_data[2, :] *= 450

    training_labels = np.zeros((30, 1))
    training_labels[10:20, :] += 1
    training_labels[20:30, :] += 2
    training_labels = training_labels.flatten()
    training_labels = list(training_labels)

    labels = student_code.nearest_neighbor_classify(training_data, training_labels, testing_data, k=1)

    gt_labels = [0, 1, 2]

    # print(labels)

    assert labels == gt_labels


def test_nearest_neighbor_classify_k():
    training_data = np.ones((5, 2))
    training_data[0, :] = [0, 0]
    training_data[1, :] = [1, 0]
    training_data[2, :] = [0.5, 1]
    training_data[3, :] = [1, 1]
    training_data[4, :] = [2, 2]

    testing_data = np.ones((1,2))
    testing_data *= 0.9

    training_labels = [0, 0, 0, 1, 1]

    labels = student_code.nearest_neighbor_classify(training_data, training_labels, testing_data, k=3)

    gt_labels = np.zeros((1, 2))

    # print(labels)

    assert (labels == gt_labels).all()


def test_kmeans_2_classes_1d_features():
    features = np.ones((10, 1))
    features[5:, :] *= 2

    centroids = student_code.kmeans(features, 2, max_iter=10)
    # print(centroids)
    gt_centroids = [[1],
                    [2]]
    gt_centroids = np.asarray(gt_centroids)
    assert gt_centroids.shape == centroids.shape
    mask = np.isin(gt_centroids, centroids)
    # print(mask)
    assert (np.all(mask) == True)


def test_kmeans_5_classes_2d_features():
    features = np.ones((60, 2))
    features[5:10, :] *= 2
    features[10:20, :] *= 12
    features[30:40, 0] *= 20
    features[30:40, 1] *= 21
    features[40:50, 1] *= 35

    centroids = student_code.kmeans(features, 5, max_iter=10)
    # print(centroids)
    gt_centroids = [[1, 1],
                    [2, 2],
                    [12, 12],
                    [20, 21],
                    [1, 35]]
    gt_centroids = np.asarray(gt_centroids)
    assert gt_centroids.shape == centroids.shape
    mask = np.isin(gt_centroids, centroids)
    # print(mask)
    assert (np.all(mask) == True)


def test_build_vocabulary_shape():
    num_images = 10
    images = []
    for ii in range(num_images):
        image = np.random.randint(0, 255, size=(50, 100)).astype('uint8')
        images.append(image)

    vocab = student_code.build_vocabulary(images, num_images)
    # print(vocab)

    assert vocab.shape == (num_images, 128)


def test_build_vocabulary_values():
    num_images = 10
    images = []
    for ii in range(num_images):
        im = np.linspace(0, 255, 640*480).astype('uint8')
        im = im.reshape((480, 640))
        image = im
        images.append(image)

    vocab = student_code.build_vocabulary(images, num_images)
    # print(vocab[0,:])

    gt_vocab = np.zeros((num_images, 128))
    gt_vocab[:, [1,9,17,25,33,41]] = 0.3

    assert np.allclose(vocab[:, :42], gt_vocab[:, :42], atol=0.05)


def test_kmeans_quantize_exact_matches():
    data = np.ones((50, 128))
    data[10:20, :] *= 20
    data[20:30, :] *= 30
    data[30:40, :] *= 40
    data[40:50, :] *= 50

    centroids = np.ones((5, 128))
    centroids[1, :] *= 20
    centroids[2, :] *= 30
    centroids[3, :] *= 40
    centroids[4, :] *= 50

    labels = student_code.kmeans_quantize(data, centroids)

    gt_labels = np.zeros((50, 1))
    gt_labels[10:20, :] += 1
    gt_labels[20:30, :] += 2
    gt_labels[30:40, :] += 3
    gt_labels[40:50, :] += 4

    assert np.equal(labels.all(), gt_labels.all())


def test_kmeans_quantize_noisy_continuous():
    data = np.ones((30, 128))
    data[:10, :] = np.random.randint(25, 75, (10, 128))
    data[10:20, :] = np.random.randint(225, 275, (10, 128))
    data[20:30, :] = np.random.randint(425, 475, (10, 128))

    centroids = np.ones((3, 128))
    centroids[0, :] *= 50
    centroids[1, :] *= 250
    centroids[2, :] *= 450

    labels = student_code.kmeans_quantize(data, centroids)

    gt_labels = np.zeros((30, 1))
    gt_labels[10:20, :] += 1
    gt_labels[20:30, :] += 2

    assert np.equal(labels.all(), gt_labels.all())


def test_get_bags_of_sifts():
    num_images = 10
    images = []
    for ii in range(num_images):
        im = np.linspace(0, 255, 640*480).astype('uint8')
        im = im.reshape((480, 640))
        image = im
        images.append(image)

    try:
        vocabulary = np.load('../proj3_unit_tests/test_data/vocab.npy')
    except:
        vocabulary = np.load('proj3_unit_tests/test_data/vocab.npy')

    vocab = student_code.get_bags_of_sifts(images, vocabulary)

    assert vocab.shape == (num_images, 50)
    assert vocab[:, 20].all() == 1.
