import glob
import os
import numpy as np

from PIL import Image
from sklearn.preprocessing import StandardScaler


def compute_mean_and_std(dir_name: str):
    '''
    Compute the mean and the standard deviation of the dataset.

    Note: convert the image in grayscale and then in [0,1] before computing mean
    and standard deviation

    Hints: use StandardScalar (check import statement)

    Args:
    -   dir_name: the path of the root dir
    Returns:
    -   mean: mean value of the dataset (np.array containing a scalar value)
    -   std: standard deviation of th dataset (np.array containing a scalar value)
    '''

    scaler = StandardScaler()
    test_folder, train_folder = glob.glob(dir_name + '/*')
    path_array = []
    cat_folders = glob.glob(train_folder + '/*')
    for cat in cat_folders:
        for img_path in glob.glob(cat + '/*.jpg'):
            path_array.append(img_path)
    cat_folders = glob.glob(test_folder + '/*')
    for cat in cat_folders:
        for img_path in glob.glob(cat + '/*.jpg'):
            path_array.append(img_path)
    for path in path_array:
        image = np.array(Image.open(path).convert('L')) / 255.0
        image = image.reshape((image.shape[0] * image.shape[1], 1))
        scaler.partial_fit(image)
    mean = scaler.mean_
    std = scaler.scale_
    return mean, std