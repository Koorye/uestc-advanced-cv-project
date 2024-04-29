'''
Script with Pytorch's dataloader class
'''

import os
import glob
import torch
import torch.utils.data as data
import torchvision

from typing import Tuple, List
from PIL import Image


class ImageLoader(data.Dataset):
    '''
    Class for data loading
    '''

    train_folder = 'train'
    test_folder = 'test'

    def __init__(self,
                root_dir: str,
                split: str = 'train',
                transform: torchvision.transforms.Compose = None):
        '''
        Init function for the class

        Args:
        - root_dir: the dir path which contains the train and test folder
        - split: 'test' or 'train' split
        - transforms: the transforms to be applied to the data
        '''
        self.root = os.path.expanduser(root_dir)
        self.transform = transform
        self.split = split

        if split == 'train':
            self.curr_folder = os.path.join(root_dir, self.train_folder)
        elif split == 'test':
            self.curr_folder = os.path.join(root_dir, self.test_folder)
        self.class_dict = self.get_classes()
        self.dataset = self.load_imagepaths_with_labels(self.class_dict)

    def load_imagepaths_with_labels(self, class_labels) -> List[Tuple[str, int]]:
        '''
        Fetches all image paths along with labels

        Args:
        -   class_labels: the class labels dictionary, with keys being the classes
            in this dataset
        Returns:
        -   list[(filepath, int)]: a list of filepaths and their class indices
        '''

        img_paths = []  # a list of (filename, class index)
        folders = glob.glob('../' + self.curr_folder + '/*')
        for f in folders:
            cat = os.path.basename(f)
            index = class_labels[cat]
            for img in glob.glob(f + '/*.jpg'):
                img_paths.append((img, index))
        return img_paths

    def get_classes(self) -> dict:
        '''
        Get the classes (which are folder names in self.curr_folder)

        Returns:
        -   Dict of class names (string) to integer labels
        '''

        classes = dict()
        folders = glob.glob('../' + self.curr_folder + '/*')
        label = 0
        for f in folders:
            cat = os.path.basename(f)
            classes[cat] = label
            label += 1
        return classes

    def load_img_from_path(self, path: str) -> Image:
        '''
        Loads the image as grayscale (using Pillow)

        Note: do not normalize the image to [0,1]

        Args:
        -   path: the path of the image
        Returns:
        -   image: grayscale image loaded using pillow (Use 'L' flag while converting using Pillow's function)
        '''
        real_path = path if path[:2] == '..' else '../' + path
        img = Image.open(real_path).convert('L')
        return img

    def __getitem__(self, index: int) -> Tuple[torch.tensor, int]:
        '''
        Fetches the item (image, label) at a given index

        Note: Do not forget to apply the transforms, if they exist

        Hint:
        1) get info from self.dataset
        2) use load_img_from_path
        3) apply transforms if valid

        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        '''

        path, class_idx = self.dataset[index]
        img = self.transform(self.load_img_from_path(path))
        return img, class_idx

    def __len__(self) -> int:
        """
        Returns the number of items in the dataset

        Returns:
            int: length of the dataset
        """
        l = len(self.dataset)
        return l