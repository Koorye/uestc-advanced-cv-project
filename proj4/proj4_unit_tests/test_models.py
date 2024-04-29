'''
Testing for models
'''

import torch
import torch.nn as nn

from collections import Counter

from proj4_code.simple_net import SimpleNet
from proj4_code.simple_net_dropout import SimpleNetDropout
from proj4_code.my_alexnet import MyAlexNet


def flatten_layers(layers):
  '''
  Keep on flattening nn.Sequential objects
  '''

  flattened_layers = list()

  recurse = False
  for elem in layers:
    if type(elem) == nn.Sequential:
      recurse = True
      flattened_layers += list(elem.children())
    else:
      flattened_layers.append(elem)

  if recurse:
    return flatten_layers(flattened_layers)

  return flattened_layers


def extract_model_layers(model: nn.Module):
  # get the CNN sequential
  layers = flatten_layers(list(model.cnn_layers.children()) +
                          list(model.fc_layers.children()))

  # generate counts of different types of layers present in the model
  layers_type = [x.__class__.__name__ for x in layers]
  layers_count = Counter(layers_type)

  # get the total number of parameters which require grad and which do not require grad
  num_params_grad = 0
  num_params_nograd = 0
  for param in model.parameters():
    if param.requires_grad:
      num_params_grad += param.numel()
    else:
      num_params_nograd += param.numel()

  return layers, layers[-1].out_features, layers_count, num_params_grad, num_params_nograd


if __name__ == '__main__':
  model1 = SimpleNet()
  print(extract_model_layers(model1))

  model2 = SimpleNetDropout()
  print(extract_model_layers(model2))

  model3 = MyAlexNet()
  print(extract_model_layers(model3))
