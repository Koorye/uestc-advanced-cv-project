from proj4_code.my_alexnet import MyAlexNet
from proj4_unit_tests.test_models import *

import numpy as np
import torch
from PIL import Image


def test_my_alexnet():
  '''
  Tests the transforms using output from disk
  '''
  this_alex_net = MyAlexNet()

  all_layers, output_dim, counter, num_params_grad, num_params_nograd = extract_model_layers(this_alex_net)

  assert output_dim == 15
  assert num_params_grad < 70000
  assert num_params_nograd > 4e7
