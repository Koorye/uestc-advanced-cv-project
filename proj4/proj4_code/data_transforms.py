'''
Contains functions with different data transforms
'''

import numpy as np
import torchvision.transforms as transforms

from typing import Tuple


def get_fundamental_transforms(inp_size: Tuple[int, int],
                               pixel_mean: np.array,
                               pixel_std: np.array) -> transforms.Compose:
    '''
    Returns the core transforms needed to feed the images to our model

    Args:
    - inp_size: tuple denoting the dimensions for input to the model
    - pixel_mean: the mean  of the raw dataset
    - pixel_std: the standard deviation of the raw dataset
    Returns:
    - fundamental_transforms: transforms.Compose with the fundamental transforms
    '''
    return transforms.Compose([
        transforms.Resize(inp_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=pixel_mean, std=pixel_std),
    ])


def get_data_augmentation_transforms(inp_size: Tuple[int, int],
                                     pixel_mean: np.array,
                                     pixel_std: np.array) -> transforms.Compose:
    '''
    Returns the data augmentation + core transforms needed to be applied on the
    train set

    Args:
    - inp_size: tuple denoting the dimensions for input to the model
    - pixel_mean: the mean  of the raw dataset
    - pixel_std: the standard deviation of the raw dataset
    Returns:
    - aug_transforms: transforms.Compose with all the transforms
    '''
    return transforms.Compose([
        transforms.RandomResizedCrop(inp_size),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=pixel_mean, std=pixel_std),
    ])
