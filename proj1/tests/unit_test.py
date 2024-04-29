#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pdb
import torch
import sys 
sys.path.append("../") 
from proj1_code.part1 import my_imfilter
from proj1_code.datasets import HybridImageDataset
from proj1_code.models import HybridImageModel, create_Gaussian_kernel
from proj1_code.utils import (
    vis_image_scales_numpy,
    im2single,
    single2im,
    load_image,
    save_image,
    write_objects_to_file
)

ROOT = Path(__file__).resolve().parent.parent  # ../..


"""
Even size kernels are not required for this project, so we exclude this test case.
"""


def get_dog_img():
	"""
	"""
	dog_img_fpath = f'{ROOT}/data/1a_dog.bmp'
	dog_img = load_image(dog_img_fpath)
	return dog_img


def test_dataloader_len():
    """
    Check dataloader __len__ for correct size (should be 5 pairs of images).
    """
    img_dir = f'{ROOT}/data'
    cut_off_file = f'{ROOT}/cutoff_frequencies.txt'
    hid = HybridImageDataset(img_dir, cut_off_file)
    assert len(hid) == 5


def test_dataloader_get_item():
	"""
	Verify that __getitem__ is implemented correctly, for the first dog/cat entry.
	"""
	img_dir = f'{ROOT}/data'
	cut_off_file = f'{ROOT}/cutoff_frequencies.txt'
	hid = HybridImageDataset(img_dir, cut_off_file)

	first_item = hid[0]
	dog_img, cat_img, cutoff = first_item

	gt_size = [3, 361, 410]
	# low frequency should be 1a_dog.bmp, high freq should be cat
	assert [dog_img.shape[i] for i in range(3)] == gt_size
	assert [cat_img.shape[i] for i in range(3)] == gt_size

	# ground truth values
	dog_img_crop = torch.tensor(
		[
			[[0.4784, 0.4745],
			[0.5255, 0.5176]],

			[[0.4627, 0.4667],
			[0.5098, 0.5137]],

			[[0.4588, 0.4706],
			[0.5059, 0.5059]]
			]
	)
	assert torch.allclose(dog_img[:,100:102,100:102], dog_img_crop, atol=1e-3)
	assert 0. < cutoff and cutoff < 1000.


def test_low_pass_filter_square_kernel():
	"""
		Allow students to use arbitrary padding types without penalty.
	"""
	dog_img = get_dog_img()
	img_h, img_w, _ = dog_img.shape
	low_pass_filter = create_Gaussian_kernel(cutoff_frequency=7)
	k_h, k_w = low_pass_filter.shape
	student_filtered_img = my_imfilter(dog_img, low_pass_filter)

	# Exclude the border pixels.
	student_filtered_img_interior = student_filtered_img[k_h:img_h-k_h, k_w:img_w-k_w]
	assert np.allclose(158332.02, student_filtered_img_interior.sum() )



def test_random_filter_nonsquare_kernel():
	"""
		Test a non-square filter (that is not a low-pass filter).
	"""
	image = np.array(range(10*15*3), dtype=np.uint8)
	image = image.reshape(10,15,3)
	image = image.astype(np.float32)
	kernel = np.array(range(3*5), dtype=np.float32).reshape(3,5) / 15
	img_h, img_w, _ = image.shape

	student_output = my_imfilter(image, kernel)

	h_center = img_h // 2
	w_center = img_w // 2

	gt_center_crop = np.array(
		[
			[[1542.0001 , 1549.     , 1556.0001 ],
			[1563.     , 1569.9999 , 1577.0001 ]],

			[[ 832.99994,  840.00006,  847.     ],
			[ 854.     ,  861.     ,  868.0001 ]]
		], dtype=np.float32
	)

	student_center_crop = student_output[h_center-1:h_center+1, w_center-1:w_center+1]
	assert np.allclose(student_center_crop, gt_center_crop, atol=1e-3)

	student_filtered_interior = student_output[1:img_h-1, 3:img_w-3,:]
	assert np.allclose( student_filtered_interior.sum(), 194196.0, atol=1e-1)


def test_random_filter_square_kernel():
	"""
		Test a square filter (that is not a low-pass filter).
	"""
	image = np.array(range(4*5*3), dtype=np.uint8)
	image = image.reshape(4,5,3)
	image = image.astype(np.float32)
	kernel = np.array(range(3*3), dtype=np.float32).reshape(3,3) / 9
	img_h, img_w, _ = image.shape

	student_output = my_imfilter(image, kernel)

	student_filtered_interior = student_output[1:img_h-1, 1:img_w-1,:]
	gt_interior_values = np.array(
		[
			[[104.     , 108.     , 112.     ],
			[116.     , 120.00001, 124.     ],
			[128.     , 132.     , 136.     ]],

			[[164.     , 168.00002, 172.     ],
			[176.     , 180.     , 184.     ],
			[188.00002, 192.     , 196.     ]]
		], dtype=np.float32
	)
	assert np.allclose(student_filtered_interior, gt_interior_values)


def verify_low_freq_sq_kernel_np(image1, kernel, low_frequencies) -> bool:
	"""
		Interactive test to be used in IPython notebook, that will print out
		test result, and return value can also be queried for success (true).

		Args:
		-	image1
		-	kernel
		-	low_frequencies

		Returns:
		-	Boolean indicating success.
	"""
	gt_image1 = load_image(f'{ROOT}/data/1a_dog.bmp')
	if not np.allclose(image1, gt_image1):
		print('Please pass in the dog image `1a_dog.bmp` as the `image1` argument.')
		return False

	img_h, img_w, _ = image1.shape
	k_h, k_w = kernel.shape
	# Exclude the border pixels.
	low_freq_interior = low_frequencies[k_h:img_h-k_h, k_w:img_w-k_w]
	correct_sum = np.allclose(158332.02, low_freq_interior.sum() )

	# ground truth values
	gt_low_freq_crop = np.array(
		[
			[[0.53500533, 0.523871  , 0.5142517 ],
			[0.5367106 , 0.526209  , 0.51830757]],

			[[0.53472066, 0.5236291 , 0.5149963 ],
			[0.5368732 , 0.5264317 , 0.5193449 ]]
		], dtype=np.float32
	)

	# H,W,C order in Numpy
	correct_crop = np.allclose(low_frequencies[100:102,100:102,:], gt_low_freq_crop, atol=1e-3)
	if correct_sum and correct_crop:
		print('Success! Low frequencies values are correct.')
		return True
	else:
		print('Low frequencies values are not correct, please double check your implementation.')
		return False

	## Purely for visualization/debugging ########
	# plt.subplot(1,2,1)
	# plt.imshow(image1)

	# plt.subplot(1,2,2)
	# plt.imshow(low_frequencies)
	# plt.show()
	##############################################


def verify_high_freq_sq_kernel_np(image2, kernel, high_frequencies) -> bool:
	"""
		Interactive test to be used in IPython notebook, that will print out
		test result, and return value can also be queried for success (true).

		Args:
		-	image2: Array representing the cat image (1b_cat.bmp)
		-	kernel: Low pass kernel (2d Gaussian)
		-	high_frequencies: High frequencies of image2 (output of high-pass filter)

		Returns:
		-	retval: Boolean indicating success.
	"""
	gt_image2 = load_image(f'{ROOT}/data/1b_cat.bmp')
	if not np.allclose(image2, gt_image2):
		print('Please pass in the cat image `1b_cat.bmp` as the `image2` argument.')
		return False

	img_h, img_w, _ = image2.shape
	k_h, k_w = kernel.shape
	# Exclude the border pixels.
	high_freq_interior = high_frequencies[k_h:img_h-k_h, k_w:img_w-k_w]
	correct_sum = np.allclose(12.029784, high_freq_interior.sum(), atol=1e-2)

	# ground truth values
	gt_high_freq_crop = np.array(
		[
			[[ 7.9535842e-03,  2.9861331e-02,  3.0958146e-02],
			[-7.6553226e-03,  2.2351682e-02,  2.7430430e-02]],

			[[ 1.5485287e-02,  3.3503681e-02,  3.0706093e-02],
			[-6.8724155e-05,  3.3921897e-02,  3.1234175e-02]]
		], dtype=np.float32
	)

	# H,W,C order in Numpy
	correct_crop = np.allclose(high_frequencies[100:102,100:102,:], gt_high_freq_crop, atol=1e-3)
	if correct_sum and correct_crop:
		print('Success! High frequencies values are correct.')
		return True
	else:
		print('High frequencies values are not correct, please double check your implementation.')
		return False

	## Purely for visualization/debugging ########
	# plt.subplot(1,2,1)
	# plt.imshow(image2)

	# plt.subplot(1,2,2)
	# high_frequencies += 0.5 # np.clip(high_frequencies, 0., 1.0)
	# plt.imshow(high_frequencies)
	# plt.show()
	##############################################


def verify_hybrid_image_np(image1, image2, kernel, hybrid_image) -> bool:
	"""
		Interactive test to be used in IPython notebook, that will print out
		test result, and return value can also be queried for success (true).

		Args:
		-	image1
		-	image2
		-	kernel
		-	hybrid_image

		Returns:
		-	Boolean indicating success.
	"""
	gt_image1 = load_image(f'{ROOT}/data/1a_dog.bmp')
	if not np.allclose(image1, gt_image1):
		print('Please pass in the dog image `1a_dog.bmp` as the `image1` argument.')
		return False

	gt_image2 = load_image(f'{ROOT}/data/1b_cat.bmp')
	if not np.allclose(image2, gt_image2):
		print('Please pass in the cat image `1b_cat.bmp` as the `image2` argument.')
		return False

	img_h, img_w, _ = image2.shape
	k_h, k_w = kernel.shape
	# Exclude the border pixels.
	hybrid_interior = hybrid_image[k_h:img_h-k_h, k_w:img_w-k_w]
	correct_sum = np.allclose(158339.52, hybrid_interior.sum() )

	# ground truth values
	gt_hybrid_crop = np.array(
		[
			[[0.5429589 , 0.55373234, 0.5452099 ],
			[0.5290553 , 0.5485607 , 0.545738  ]],

			[[0.55020595, 0.55713284, 0.5457024 ],
			[0.5368045 , 0.5603536 , 0.5505791 ]]
		], dtype=np.float32
	)

	# H,W,C order in Numpy
	correct_crop = np.allclose(hybrid_image[100:102,100:102,:], gt_hybrid_crop, atol=1e-3)
	if correct_sum and correct_crop:
		print('Success! Hybrid image values are correct.')
		return True
	else:
		print('Hybrid image values are not correct, please double check your implementation.')
		return False


	## Purely for debugging/visualization ##
	# plt.imshow(hybrid_image)
	# plt.show()
	########################################


def verify_gaussian_kernel(kernel, cutoff_frequency) -> bool:
	"""
		Interactive test to be used in IPython notebook, that will print out
		test result, and return value can also be queried for success (true).

		Args:
		-	kernel
		-	cutoff_frequency

		Returns:
		-	Boolean indicating success.
	"""
	if cutoff_frequency != 7:
		print('Please change the cutoff_frequency back to 7 and rerun this test')
		return False
	if kernel.shape != (29,29):
		print('The kernel is not the correct size')
		return False

	kernel_h, kernel_w = kernel.shape
	gt_kernel_crop = np.array(
		[
			[0.00323564, 0.00333623, 0.00337044, 0.00333623],
			[0.00333623, 0.00343993, 0.00347522, 0.00343993],
			[0.00337044, 0.00347522, 0.00351086, 0.00347522],
			[0.00333623, 0.00343993, 0.00347522, 0.00343993]
		]
	)

	h_center = kernel_h // 2
	w_center = kernel_w // 2
	student_kernel_crop = kernel[h_center-2:h_center+2, w_center-2:w_center+2]

	correct_crop = np.allclose(gt_kernel_crop, student_kernel_crop, atol=1e-7)
	correct_sum = np.allclose(kernel.sum(), 1.0, atol=1e-3)
	correct_vals = correct_crop and correct_sum

	if correct_vals:
		print('Success -- kernel values are correct.')
		return True
	else:
		print('Kernel values are not correct.')
		return False


def test_pytorch_low_pass_filter_square_kernel():
	"""
	Test the low pass filter, but not the output of the forward() pass.
	"""
	hi_model = HybridImageModel()
	img_dir = f'{ROOT}/data'
	cut_off_file = f'{ROOT}/cutoff_frequencies_temp.txt'

	# Dump to a file
	cutoff_freqs = [7, 7, 7, 7, 7]
	write_objects_to_file(fpath=cut_off_file, obj_list=cutoff_freqs)
	hi_dataset = HybridImageDataset(img_dir, cut_off_file)

	# should be the dog image
	img_a, img_b, cutoff_freq = hi_dataset[0]
	# turn CHW into NCHW
	img_a = img_a.unsqueeze(0)

	hi_model.n_channels = 3
	kernel = hi_model.get_kernel(cutoff_freq)
	pytorch_low_freq = hi_model.low_pass(img_a, kernel)

	assert list(pytorch_low_freq.shape) == [1,3,361,410]
	assert isinstance(pytorch_low_freq, torch.Tensor)

	# crop from pytorch_output[:,:,20:22,20:22]
	gt_crop = torch.tensor(
		[
			[
				[[0.7941, 0.7989],
				[0.7906, 0.7953]],

				[[0.9031, 0.9064],
				[0.9021, 0.9052]],

				[[0.9152, 0.9173],
				[0.9168, 0.9187]]
			]
		], dtype=torch.float32
	)
	assert torch.allclose(pytorch_low_freq[:,:,20:22,20:22], gt_crop, atol=1e-3)

	# ground truth element sum
	assert np.allclose( pytorch_low_freq.numpy().sum(), 209926.3481)


def verify_low_freq_sq_kernel_pytorch(image_a, model, cutoff_freq, low_frequencies) -> bool:
	"""
		Test the output of the forward pass.

		Args:
		-	image_a
		-	model
		-	cutoff_freq
		-	low_frequencies

		Returns:
		-	None
	"""
	if not isinstance(cutoff_freq, torch.Tensor) or not torch.allclose(cutoff_freq, torch.Tensor([7])):
		print('Please pass a Pytorch tensor containing `7` as the cutoff frequency.')
		return False

	img_a_val_sum = float(image_a.sum())
	if not np.allclose(img_a_val_sum, 215154.9531):
		print('Please pass in the dog image `1a_dog.bmp` as the `image_a` argument.')
		return False

	gt_low_freq_crop = torch.tensor(
		[
			[[0.5350, 0.5367],
			[0.5347, 0.5369]],

			[[0.5239, 0.5262],
			[0.5236, 0.5264]],

			[[0.5143, 0.5183],
			[0.5150, 0.5193]]
		]
	)
	correct_crop = torch.allclose(gt_low_freq_crop, low_frequencies[0,:,100:102,100:102], atol=1e-3)

	img_h = image_a.shape[2]
	img_w = image_a.shape[3]
	kernel = model.get_kernel(int(cutoff_freq))
	if not isinstance(kernel, torch.Tensor):
		print('Kernel is not a torch tensor')
		return False

	gt_kernel_sz_list = [3,1,29,29]
	kernel_sz_list = [int(val) for val in kernel.shape]

	if gt_kernel_sz_list != kernel_sz_list:
		print('Kernel is not the correct size')
		return False

	k_h = kernel.shape[2]
	k_w = kernel.shape[3]


	# Exclude the border pixels.
	low_freq_interior = low_frequencies[0, :, k_h:img_h-k_h, k_w:img_w-k_w]
	correct_sum = np.allclose(158332.06, float(low_freq_interior.sum()), atol=1)

	if correct_sum and correct_crop:
		print('Success! Pytorch low frequencies values are correct.')
		return True
	else:
		print('Pytorch low frequencies values are not correct, please double check your implementation.')
		return False


def verify_high_freq_sq_kernel_pytorch(image_b, model, cutoff_freq, high_frequencies) -> bool:
	"""
		Test the output of the forward pass.

		Args:
		-	image_b
		-	model
		-	cutoff_freq
		-	high_frequencies

		Returns:
		-	None
	"""
	if not isinstance(cutoff_freq, torch.Tensor) or not torch.allclose(cutoff_freq, torch.Tensor([7])):
		print('Please pass a Pytorch tensor containing `7` as the cutoff frequency.')
		return False

	img_b_val_sum = float(image_b.sum())
	if not np.allclose(img_b_val_sum, 230960.1875, atol=5.0):
		print('Please pass in the cat image `1b_cat.bmp` as the `image_b` argument.')
		return False

	gt_high_freq_crop = torch.tensor(
		[
			[[ 7.9527e-03, -7.6560e-03],
			[ 1.5484e-02, -6.9082e-05]],

			[[ 2.9861e-02,  2.2352e-02],
			[ 3.3504e-02,  3.3922e-02]],

			[[ 3.0958e-02,  2.7430e-02],
			[ 3.0706e-02,  3.1234e-02]]
		]
	)
	correct_crop = torch.allclose(gt_high_freq_crop, high_frequencies[0,:,100:102,100:102], atol=1e-3)

	img_h = image_b.shape[2]
	img_w = image_b.shape[3]
	kernel = model.get_kernel(int(cutoff_freq))
	if not isinstance(kernel, torch.Tensor):
		print('Kernel is not a torch tensor')
		return False

	gt_kernel_sz_list = [3,1,29,29]
	kernel_sz_list = [int(val) for val in kernel.shape]

	if gt_kernel_sz_list != kernel_sz_list:
		print('Kernel is not the correct size')
		return False

	k_h = kernel.shape[2]
	k_w = kernel.shape[3]

	# Exclude the border pixels.
	high_freq_interior = high_frequencies[0, :, k_h:img_h-k_h, k_w:img_w-k_w]
	correct_sum = np.allclose(12.012651, float(high_freq_interior.sum()), atol=1e-1)

	if correct_sum and correct_crop:
		print('Success! Pytorch high frequencies values are correct.')
		return True
	else:
		print('Pytorch high frequencies values are not correct, please double check your implementation.')
		return False


def verify_hybrid_image_pytorch(image_a, image_b, model, cutoff_freq, hybrid_image) -> bool:
	"""
		Test the output of the forward pass.

		Args:
		-	image_a
		-	image_b
		-	model
		-	cutoff_freq
		-	hybrid_image

		Returns:
		-	None
	"""
	_, _, img_h, img_w = image_b.shape
	kernel = model.get_kernel(int(cutoff_freq))
	_, _, k_h, k_w = kernel.shape

	# Exclude the border pixels.
	hybrid_interior = hybrid_image[0, :, k_h:img_h-k_h, k_w:img_w-k_w]
	correct_sum = np.allclose(158339.5469, hybrid_interior.sum(), atol=1e-2)

	# ground truth values
	gt_hybrid_crop = torch.tensor(
		[
			[[0.5430, 0.5291],
			[0.5502, 0.5368]],

			[[0.5537, 0.5486],
			[0.5571, 0.5604]],

			[[0.5452, 0.5457],
			[0.5457, 0.5506]]
		]
	)
	# H,W,C order in Numpy
	correct_crop = torch.allclose(hybrid_image[0,:,100:102,100:102], gt_hybrid_crop, atol=1e-3)
	if correct_sum and correct_crop:
		print('Success! Pytorch hybrid image values are correct.')
		return True
	else:
		print('Pytorch hybrid image values are not correct, please double check your implementation.')
		return False
