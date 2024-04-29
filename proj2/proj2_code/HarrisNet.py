#!/usr/bin/python3

from torch import nn
import torch
from typing import Tuple
import numpy as np
import torch.nn.functional as F


from proj2_code.torch_layer_utils import (
    get_sobel_xy_parameters,
    get_gaussian_kernel,
    ImageGradientsLayer
)

"""
Authors: Patsorn Sangkloy, Vijay Upadhya, John Lambert, Cusuh Ham,
Frank Dellaert, September 2019.
"""


class HarrisNet(nn.Module):
    """
    Implement Harris corner detector (See Szeliski 4.1.1) in pytorch by
    sequentially stacking several layers together.

    Your task is to implement the combination of pytorch module custom layers
    to perform Harris Corner detector.

    Recall that R = det(M) - alpha(trace(M))^2
    where M = [S_xx S_xy;
               S_xy  S_yy],
          S_xx = Gk * I_xx
          S_yy = Gk * I_yy
          S_xy  = Gk * I_xy,
    and * is a convolutional operation over a Gaussian kernel of size (k, k).
    (You can verify that this is equivalent to taking a (Gaussian) weighted sum
    over the window of size (k, k), see how convolutional operation works here:
    http://cs231n.github.io/convolutional-networks/)

    Ix, Iy are simply image derivatives in x and y directions, respectively.

    You may find the Pytorch function nn.Conv2d() helpful here.
    """

    def __init__(self):
        """
        Create a nn.Sequential() network, using 5 specific layers (not in this
        order):
          - SecondMomentMatrixLayer: Compute S_xx, S_yy and S_xy, the output is
            a tensor of size (num_image, 3, width, height)
          - ImageGradientsLayer: Compute image gradients Ix Iy. Can be
            approximated by convolving with Sobel filter.
          - NMSLayer: Perform nonmaximum suppression, the output is a tensor of
            size (num_image, 1, width, height)
          - ChannelProductLayer: Compute I_xx, I_yy and I_xy, the output is a
            tensor of size (num_image, 3, width, height)
          - CornerResponseLayer: Compute R matrix, the output is a tensor of
            size (num_image, 1, width, height)

        To help get you started, we give you the ImageGradientsLayer layer to
        compute Ix and Iy. You will need to implement all the other layers. You
        will need to combine all the layers together using nn.Sequential, where
        the output of one layer will be the input to the next layer, and so on
        (see HarrisNet diagram). You'll also need to find the right order since
        the above layer list is not sorted ;)

        Args:
        -   None

        Returns:
        -   None
        """
        super(HarrisNet, self).__init__()

        image_gradients_layer = ImageGradientsLayer()
        channel_product_layer = ChannelProductLayer()
        second_moment_matrix_layer = SecondMomentMatrixLayer()
        corner_response_layer = CornerResponseLayer()
        nms_layer = NMSLayer()

        model = torch.nn.Sequential(
            image_gradients_layer,
            channel_product_layer,
            second_moment_matrix_layer,
            corner_response_layer,
            nms_layer,
        )
        self.net = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass of HarrisNet network. We will only test with 1
        image at a time, and the input image will have a single channel.

        Args:
        -   x: input Tensor of shape (num_image, channel, height, width)

        Returns:
        -   output: output of HarrisNet network,
            (num_image, 1, height, width) tensor

        """
        assert x.dim() == 4, \
            "Input should have 4 dimensions. Was {}".format(x.dim())

        return self.net(x)


class ChannelProductLayer(torch.nn.Module):
    """
    ChannelProductLayer: Compute I_xx, I_yy and I_xy,

    The output is a tensor of size (num_image, 3, height, width), each channel
    representing I_xx, I_yy and I_xy respectively.
    """
    def __init__(self):
        super(ChannelProductLayer, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The input x here is the output of the previous layer, which is of size
        (num_image x 2 x width x height) for Ix and Iy.

        Args:
        -   x: input tensor of size (num_image, channel, height, width)

        Returns:
        -   output: output of HarrisNet network, tensor of size
            (num_image, 3, height, width) for I_xx, I_yy and I_xy.

        HINT: you may find torch.cat(), torch.mul() useful here
        """

        num_image, c, height, width = x.shape
        output = torch.zeros([num_image, 3, height, width])

        for i in range(num_image):
            image_gx = x[i][0]
            image_gy = x[i][1]
            I_xx = torch.mul(image_gx, image_gx).unsqueeze(0)
            I_yy = torch.mul(image_gy, image_gy).unsqueeze(0)
            I_xy = torch.mul(image_gx, image_gy).unsqueeze(0)
            output[i] = torch.cat((I_xx, I_yy, I_xy))
        return output


class SecondMomentMatrixLayer(torch.nn.Module):
    """
    SecondMomentMatrixLayer: Given a 3-channel image I_xx, I_xy, I_yy, then
    compute S_xx, S_yy and S_xy.

    The output is a tensor of size (num_image, 3, height, width), each channel
    representing S_xx, S_yy and S_xy, respectively

    """
    def __init__(self, ksize: torch.Tensor = 7, sigma: torch.Tensor = 5):
        """
        You may find get_gaussian_kernel() useful. You must use a Gaussian
        kernel with filter size `ksize` and standard deviation `sigma`. After
        you pass the unit tests, feel free to experiment with other values.

        Args:
        -   None

        Returns:
        -   None
        """
        super(SecondMomentMatrixLayer, self).__init__()
        self.ksize = ksize
        self.sigma = sigma
        self.kernel = get_gaussian_kernel(self.ksize, self.sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The input x here is the output of previous layer, which is of size
        (num_image, 3, width, height) for I_xx and I_yy and I_xy.

        Args:
        -   x: input tensor of size (num_image, channel, height, width)

        Returns:
        -   output: output of HarrisNet network, tensor of size
            (num_image, 3, height, width) for S_xx, S_yy and S_xy

        HINT:
        - You can either use your own implementation from project 1 to get the
        Gaussian kernel, OR reimplement it in get_gaussian_kernel().
        """

        num_image, c, height, width = x.shape
        output = torch.zeros([num_image, 3, height, width])
        kernel = self.kernel
        k = kernel.shape[0]
        kernel = torch.reshape(kernel, (1, 1, k, k))

        for i in range(num_image):
            I_xx = x[i][0].unsqueeze(0).unsqueeze(0)
            I_yy = x[i][1].unsqueeze(0).unsqueeze(0)
            I_xy = x[i][2].unsqueeze(0).unsqueeze(0)

            S_xx = F.conv2d(input=I_xx,
                            weight=kernel.float(),
                            padding=k//2)
            S_yy = F.conv2d(input=I_yy,
                            weight=kernel.float(),
                            padding=k//2)
            S_xy = F.conv2d(input=I_xy,
                            weight=kernel.float(),
                            padding=k//2)

            output[i][0] = S_xx
            output[i][1] = S_yy
            output[i][2] = S_xy
        return output


class CornerResponseLayer(torch.nn.Module):
    """
    Compute R matrix.

    The output is a tensor of size (num_image, channel, height, width),
    represent corner score R

    HINT:
    - For matrix A = [a b;
                      c d],
      det(A) = ad-bc, trace(A) = a+d
    """
    def __init__(self, alpha: int=0.05):
        """
        Don't modify this __init__ function!
        """
        super(CornerResponseLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass to compute corner score R

        Args:
        -   x: input tensor of size (num_image, channel, height, width)

        Returns:
        -   output: the output is a tensor of size
            (num_image, 1, height, width), each representing harris corner
            score R

        You may find torch.mul() useful here.
        """
        num_image, c, height, width = x.shape
        output = torch.zeros([num_image, 1, height, width])
        for i in range(num_image):
            S_xx = x[i][0]
            S_yy = x[i][1]
            S_xy = x[i][2]
            det = torch.mul(S_xx, S_yy) - torch.mul(S_xy, S_xy)
            trace = S_xx + S_yy
            R = det - self.alpha * (trace ** 2)
            output[i] = R
        return output


class NMSLayer(torch.nn.Module):
    """
    NMSLayer: Perform non-maximum suppression,

    the output is a tensor of size (num_image, 1, height, width),

    HINT: One simple way to do non-maximum suppression is to simply pick a
    local maximum over some window size (u, v). This can be achieved using
    nn.MaxPool2d. Note that this would give us all local maxima even when they
    have a really low score compare to other local maxima. It might be useful
    to threshold out low value score before doing the pooling (torch.median
    might be useful here).

    You will definitely need to understand how nn.MaxPool2d works in order to
    utilize it, see https://pytorch.org/docs/stable/nn.html#maxpool2d
    """
    def __init__(self):
        super(NMSLayer, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Threshold globally everything below the median to zero, and then
        MaxPool over a 7x7 kernel. This will fill every entry in the subgrids
        with the maximum nearby value. Binarize the image according to
        locations that are equal to their maximum, and return this binary
        image, multiplied with the cornerness response values. We'll be testing
        only 1 image at a time. Input and output will be single channel images.

        Args:
        -   x: input tensor of size (num_image, channel, height, width)

        Returns:
        -   output: the output is a tensor of size
            (num_image, 1, height, width), each representing harris corner
            score R

        (Potentially) useful functions: nn.MaxPool2d, torch.where(), torch.median()
        """
        num_image, c, height, width = x.shape
        zero_tensor = torch.zeros([c, height, width])
        output = torch.zeros([num_image, 1, height, width])

        for i in range(num_image):
            image = x[i]
            thresh = torch.median(image)
            image = torch.where(image >= thresh, image, zero_tensor)
            pool = torch.nn.MaxPool2d(kernel_size=7,
                                      padding=7//2,
                                      stride=1)
            maximums = pool(image)
            output[i] = torch.where(image == maximums,
                                    image,
                                    zero_tensor)
        return output


def get_interest_points(image: torch.Tensor, num_points: int = 4500) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Function to return top most N x,y points with the highest confident corner
    score. Note that the return type should be a tensor. Also make sure to
    sort them in order of confidence!

    (Potentially) useful functions: torch.nonzero, torch.masked_select,
    torch.argsort

    Args:
    -   image: A tensor of shape (b,c,m,n). We will provide an image of
        (c = 1) for grayscale image.

    Returns:
    -   x: A tensor array of shape (N,) containing x-coordinates of
        interest points
    -   y: A tensor array of shape (N,) containing y-coordinates of
        interest points
    -   confidences (optional): tensor array of dim (N,) containing the
        strength of each interest point
    """

    # We initialize the Harris detector here, you'll need to implement the
    # HarrisNet() class
    harris_detector = HarrisNet()

    # The output of the detector is an R matrix of the same size as image,
    # indicating the corner score of each pixel. After non-maximum suppression,
    # most of R will be 0.
    R = harris_detector(image)

    useless_1, useless_2, y, x = torch.nonzero(R, as_tuple=True)
    confidences = R[0, 0, y, x]
    indices = torch.argsort(confidences, descending=True)
    x = x[indices]
    y = y[indices]
    confidences = torch.tensor(sorted(confidences, reverse=True))

    length = x.shape[0]
    if length > num_points:
        x = x[:num_points]
        y = y[:num_points]
        confidences = confidences[:num_points]
    y, x, confidences = remove_border_vals(R, y, x, confidences)
    return x, y, confidences


def remove_border_vals(img, x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Remove interest points that are too close to a border to allow SIFTfeature
    extraction. Make sure you remove all points where a 16x16 window around
    that point cannot be formed.

    Args:
    -   x: Torch tensor of shape (M,)
    -   y: Torch tensor of shape (M,)
    -   c: Torch tensor of shape (M,)

    Returns:
    -   x: Torch tensor of shape (N,), where N <= M (less than or equal after pruning)
    -   y: Torch tensor of shape (N,)
    -   c: Torch tensor of shape (N,)
    """
    useless_1, useless_2, height, width = img.shape
    keeped = []

    for i in range(x.shape[0]):
        if x[i] - 8 > 0 and x[i] + 8 < height and y[i] - 8 > 0 and y[i] + 8 < width:
            keeped.append(i)
    length = x.shape[0]
    x = torch.tensor([x[i] for i in range(length) if i in keeped])
    y = torch.tensor([y[i] for i in range(length) if i in keeped])
    c = torch.tensor([c[i] for i in range(length) if i in keeped])
    return x, y, c
