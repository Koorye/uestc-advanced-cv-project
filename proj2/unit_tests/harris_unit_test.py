#!/usr/bin/python3

import torch

from proj2_code.HarrisNet import (
  ImageGradientsLayer, 
  ChannelProductLayer,
  SecondMomentMatrixLayer, 
  CornerResponseLayer, 
  NMSLayer, 
  HarrisNet, 
  get_interest_points
)


def verify(function) -> str:
  """ Will indicate with a print statement whether assertions passed or failed
    within function argument call.

    Args:
    - function: Python function object

    Returns:
    - string
  """
  try:
    function()
    return "\x1b[32m\"Correct\"\x1b[0m"
  except AssertionError:
    return "\x1b[31m\"Wrong\"\x1b[0m"


def test_HarrisNet():
  """
  Tests HarrisNet as a corner detector. 
  """
  #Here we test with the dummy image, the HarrisNet should return corner score at (1,1)
  dummy_image = torch.tensor(
    [
      [1., 0., 1.],
      [0., 1., 0.],
      [1., 0., 1.]
    ]).unsqueeze(0).unsqueeze(0).float()
  harris_detector = HarrisNet()
  output = harris_detector(dummy_image)
  assert output.shape == dummy_image.shape, "the shape of the output should be the same as the input image"

  output = output / torch.max(output) #normalize them to 1
  assert output[:,:,1,1] == 1 #the most cornerness point should be at (1,1) with the value of 1


def test_get_interest_points():
  """
  Tests that get_interest_points function can get the correct coordinate. 
  """    
  dummy_image = torch.tensor(
    [
      [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
      [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
      [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
      [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
      [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
      [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
      [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
      [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
      [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]
    ]).unsqueeze(0).unsqueeze(0).float()

  x, y, confidence = get_interest_points(dummy_image)
  xy = [(x[i],y[i]) for i in range(len(x))]
  assert (9,9) in xy #(9,9) must be in the interest points


def test_ImageGradientsLayer():
  """
  Sanity check, test ImageGradientsLayer output with ground truth (gt).
  """
  x = torch.tensor(
    [
      [2, 8, 0],
      [2, 4, 4],
      [1, 5, 5]
    ]).unsqueeze(0).unsqueeze(0).float()

  Ix_gt = torch.tensor(
    [
      [ 20.,  -2., -20.],
      [ 21.,   6., -21.],
      [ 14.,  10., -14.]
    ])

  Iy_gt = torch.tensor(
    [
      [  8.,  14.,  12.],
      [ -5.,  -2.,   7.],
      [ -8., -14., -12.]
    ])

  img_grad = ImageGradientsLayer()
  out = img_grad(x)
  Ix = out[:,0,:,:]
  Iy = out[:,1,:,:]
  assert torch.allclose(Ix_gt, Ix.unsqueeze(0), atol=1) and torch.allclose(Iy_gt, Iy.unsqueeze(0), atol=1) 


def test_SecondMomentMatrixLayer():
  """
  test SecondMomentMatrixLayer. Convert Tensor of shape (1, 3, 3, 3) to (1, 3, 3, 3).
  """
  x = torch.tensor(
    [
      [
        [[16.,  9.,  0.],
        [ 0.,  9.,  4.],
        [16.,  4.,  4.]],

        [[ 4.,  4.,  0.],
        [ 4.,  1.,  0.],
        [ 9.,  4.,  1.]],

        [[ 8.,  6.,  0.],
        [ 0.,  3.,  0.],
        [12.,  4.,  2.]]
      ]
    ]).float()

  #sanity check, ksize=1 sigma =1, output = input
  secondmm = SecondMomentMatrixLayer(ksize=1, sigma=1)
  out = secondmm(x)
  assert torch.all(x==out)

  #case 2: ksize =3 sigma = 3
  secondmm = SecondMomentMatrixLayer(ksize=3, sigma=3)
  out = secondmm(x)
  gt = torch.tensor(
    [
      [
        [[3.8941, 4.3319, 2.4334],
        [6.0285, 6.8509, 3.3397],
        [3.3286, 4.1865, 2.3461]],

        [[1.4902, 1.4718, 0.5594],
        [2.9178, 2.9749, 1.0822],
        [2.0880, 2.1505, 0.6790]],

        [[1.9562, 1.9616, 0.9997],
        [3.6715, 3.8438, 1.6355],
        [2.2083, 2.4012, 1.0126]]
      ]
    ])

  assert torch.allclose(out,gt,rtol=1e-04)


def test_ChannelProductLayer():
  """
  test ChannelProductLayer. Convert tensor of shape (1, 2, 3, 3) to 
  tensor of shape (1, 3, 3, 3).
  """
  x = torch.tensor(
    [
      [[4, 3, 0],
      [0, 3, 2],
      [4, 2, 2]],

      [[2, 2, 0],
      [2, 1, 0],
      [3, 2, 1]]
    ]).unsqueeze(0).float()

  #sanity check
  cproduct = ChannelProductLayer()
  out = cproduct(x)

  Ix2 = torch.tensor(
    [
      [16.,  9.,  0.],
      [ 0.,  9.,  4.],
      [16.,  4.,  4.]
    ]).unsqueeze(0).float()

  Iy2 = torch.tensor(
    [
      [ 4.,  4.,  0.],
      [ 4.,  1.,  0.],
      [ 9.,  4.,  1.]
    ]).unsqueeze(0).float()

  IxIy = torch.tensor(
    [
      [ 8.,  6.,  0.],
      [ 0.,  3.,  0.],
      [12.,  4.,  2.]
    ]).unsqueeze(0).float()

  assert torch.all(Ix2==out[:,0,:,:])
  assert torch.all(Iy2==out[:,1,:,:])
  assert torch.all(IxIy==out[:,2,:,:])

  
def test_CornerResponseLayer():
  """
  test CornerResponseLayer. Convert tensor of shape (1, 3, 3, 3) to (1, 1, 3, 3)
  """

  S = torch.tensor(
    [
      [[4, 3, 0],
      [0, 3, 2],  #S_xx
      [4, 2, 2]],

      [[2, 2, 0],
      [2, 1, 0], #S_yy
      [3, 2, 1]],

      [[3, 0, 3],
      [4, 4, 1], #S_xy
      [2, 0, 1]]
    ]).unsqueeze(0).float()

  compute_score = CornerResponseLayer(alpha=0.05) #test at a specific alpha value
  R = compute_score(S)
  R_gt = torch.tensor(
    [
      [
        [[ -2.8000,   4.7500,  -9.0000],
        [-16.2000, -13.8000,  -1.2000],
        [  5.5500,   3.2000,   0.5500]]
      ]
    ])
  assert torch.allclose(R, R_gt.unsqueeze(0),rtol=1e-04)


def test_NMSLayer():
  """
  test NMSLayer. Convert tensor (1, 1, 3, 3) to (1, 1, 3, 3).
  """
  R = torch.tensor(
    [
      [1, 4, 4],
      [0, 1, 1],
      [2, 2, 2]
    ]).unsqueeze(0).unsqueeze(0).float()

  nms = NMSLayer()
  R_nms = nms(R)

  gt = torch.tensor(
    [
      [
        [[0., 4., 4.],
        [0., 0., 0.],
        [0., 0., 0.]]
      ]
    ])
  assert R_nms.shape == torch.Size([1, 1, 3, 3]), "Incorrect size, please check your implementation"
  assert torch.allclose(R_nms,gt,rtol=1e-4)



