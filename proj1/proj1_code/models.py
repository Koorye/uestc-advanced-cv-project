#!/usr/bin/python3

"""
PyTorch tutorial on constructing neural networks:
https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from proj1_code.part1 import create_Gaussian_kernel


class HybridImageModel(nn.Module):
    def __init__(self):
        """
        Initializes an instance of the HybridImageModel class.
        """
        super(HybridImageModel, self).__init__()

    def get_kernel(self, cutoff_frequency: int) -> torch.Tensor:
        """
        Returns a Gaussian kernel using the specified cutoff frequency.

        PyTorch requires the kernel to be of a particular shape in order to apply
        it to an image. Specifically, the kernel needs to be of shape (c, 1, k, k)
        where c is the # channels in the image. Start by getting a 2D Gaussian
        kernel using your implementation from Part 1, which will be of shape
        (k, k). Then, let's say you have an RGB image, you will need to turn this
        into a Tensor of shape (3, 1, k, k) by stacking the Gaussian kernel 3
        times.

        Args
        - cutoff_frequency: int specifying cutoff_frequency
        Returns
        - kernel: Tensor of shape (c, 1, k, k) where c is # channels

        HINTS:
        - You will use the create_Gaussian_kernel() function from part1.py in this
        function.
        - Since the # channels may differ across each image in the dataset, make
        sure you don't hardcode the dimensions you reshape the kernel to. There
        is a variable defined in this class to give you channel information.
        - You can use np.reshape() to change the dimensions of a numpy array.
        - You can use np.tile() to repeat a numpy array along specified axes.
        - You can use torch.Tensor() to convert numpy arrays to torch Tensors.
        """

        kernel = create_Gaussian_kernel(cutoff_frequency)
        c = self.n_channels
        k = kernel.shape[0]

        kernel = np.reshape(kernel, (1, k ** 2))
        kernel = np.tile(kernel, c)
        kernel = np.reshape(kernel, (c, 1, k, k))
        kernel = torch.Tensor(kernel)
        return kernel

    def low_pass(self, x, kernel):
        """
        Applies low pass filter to the input image.

        Args:
        - x: Tensor of shape (b, c, m, n) where b is batch size
        - kernel: low pass filter to be applied to the image
        Returns:
        - filtered_image: Tensor of shape (b, c, m, n)

        HINT:
        - You should use the 2d convolution operator from torch.nn.functional.
        - Make sure to pad the image appropriately (it's a parameter to the
        convolution function you should use here!).
        - Pass self.n_channels as the value to the "groups" parameter of the
        convolution function. This represents the # of channels that the filter
        will be applied to.
        """

        k = kernel.shape[2]
        filtered_image = F.conv2d(input=x.float(),
                                  weight=kernel,
                                  padding=k // 2,
                                  groups=self.n_channels)
        return filtered_image

    def forward(self, image1, image2, cutoff_frequency):
        """
        Takes two images and creates a hybrid image. Returns the low frequency
        content of image1, the high frequency content of image 2, and the hybrid
        image.

        Args
        - image1: Tensor of shape (b, c, m, n)
        - image2: Tensor of shape (b, c, m, n)
        - cutoff_frequency: Tensor of shape (b)
        Returns:
        - low_frequencies: Tensor of shape (b, c, m, n)
        - high_frequencies: Tensor of shape (b, c, m, n)
        - hybrid_image: Tensor of shape (b, c, m, n)

        HINTS:
        - You will use the get_kernel() function and your low_pass() function in
        this function.
        - Similar to Part 1, you can get just the high frequency content of an
        image by removing its low frequency content.
        - Don't forget to make sure to clip the pixel values >=0 and <=1. You can
        use torch.clamp().
        - If you want to use images with different dimensions, you should resize
        them in the HybridImageDataset class using torchvision.transforms.
        """
        self.n_channels = image1.shape[1]

        kernel = self.get_kernel(int(cutoff_frequency.item()))

        low_freq_1 = self.low_pass(image1, kernel)
        low_freq_2 = self.low_pass(image2, kernel)

        high_freq_2 = image2.float() - low_freq_2

        hybrid_image = torch.clamp(low_freq_1 + high_freq_2, 0, 1)
        low_frequencies = low_freq_1
        high_frequencies = high_freq_2
        return low_frequencies, high_frequencies, hybrid_image
