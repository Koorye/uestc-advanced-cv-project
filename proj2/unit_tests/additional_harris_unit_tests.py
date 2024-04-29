#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pdb
import torch
import torchvision.transforms as transforms

from proj2_code.utils import (
    load_image,
    rgb2gray,
    PIL_resize,
	show_interest_points
)

from proj2_code.HarrisNet import get_interest_points


ROOT = Path(__file__).resolve().parent.parent  # ../..


def test_find_single_valid_corner():
	"""
	Given two cross patterns in an image (middle right and bottom left), identify the only valid interest point.
	The pixel at which the cross pattern at the bottom-left is centered will not allow a valid 16x16 grid.
	"""
	img = torch.tensor(
		[
			[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
			[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
			[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
			[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
			[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
			[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
			[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
			[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
			[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
			[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
			[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
			[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
			[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
			[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
			[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
			[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
			[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
			[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
			[0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
			[0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
			[1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
			[0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
			[0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
			[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
			[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
		]).unsqueeze(0).unsqueeze(0).float()

	x, y, confidence = get_interest_points(img * 250)
	assert x[0] == 16
	assert y[0] == 10


def test_top10_notredame1a():
	"""
	Verify that we can retrieve the 10 pixels with highest cornerness score (that do not lie
	at invalid locations close to image borders).
	"""
	image1 = load_image(f'{ROOT}/data/1a_notredame.jpg')

	scale_factor = 0.5
	image1 = PIL_resize(image1, (int(image1.shape[1]*scale_factor), int(image1.shape[0]*scale_factor)))
	image1_bw = rgb2gray(image1)
	tensor_type = torch.FloatTensor
	torch.set_default_tensor_type(tensor_type)
	to_tensor = transforms.ToTensor()
	image_input1 = to_tensor(image1_bw).unsqueeze(0)

	x1, y1, _ = get_interest_points(image_input1, num_points=12)
	x1, y1 = x1.detach().numpy(), y1.detach().numpy()

	gt_x1 = np.array([411, 395, 346, 364, 379, 467, 349, 468, 656, 404])
	gt_y1 = np.array([408, 410, 439, 411, 410, 159, 427, 170, 156, 400])

	# only checking the first 10/12, in case you pruned borders before or after confidence pruning
	print(x1)
	print(y1)
	assert np.allclose(x1[:10], gt_x1)
	assert np.allclose(y1[:10], gt_y1)

	# -- Purely for visualization/debugging: visualize the interest points ---
	# c1 = show_interest_points(image1, x1, y1)
	# plt.figure()
	# plt.imshow(c1)
	# plt.show()
	# ------------------------------------------------------------------------


if __name__ == '__main__':
	test_find_single_valid_corner()
	test_top10_notredame1a()



