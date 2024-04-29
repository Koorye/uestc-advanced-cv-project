import numpy as np
from numpy import transpose
import matplotlib
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image

from utils import load_image, save_image, vis_image_scales_numpy
from part1 import my_imfilter, create_hybrid_image, create_Gaussian_kernel

image_a = Image.open('../data/1a_dog.bmp')
pixels = np.array(image_a) / 255.0
print(pixels.shape)

transform = transforms.Compose([transforms.ToTensor()])
image_a = transform(pixels)
print(image_a.shape)
