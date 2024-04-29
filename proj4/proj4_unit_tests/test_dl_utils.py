import torch.nn as nn
import torch

from proj4_code.dl_utils import predict_labels, compute_loss


class TestModel(nn.Module):
  def __init__(self):
    super(TestModel, self).__init__()
    self.net = nn.Linear(5, 5, bias=False)

    self.net.weight = nn.Parameter(
        torch.arange(25, dtype=torch.float32).reshape(5, 5)-12)

    self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')

  def forward(self, x):
    return self.net(x)


def test_predict_labels():
  '''
  Test the label prediction logic on a dummy net
  '''

  test_net = TestModel()

  x = torch.FloatTensor([+1.4, -1.4, -0.7, 2.3, 0.3]).reshape(1, -1)

  labels = predict_labels(test_net, x)
  assert labels.item() == 4


def test_compute_loss():
  '''
  Test the loss computation on a dummy data
  '''

  test_net = TestModel()

  x = torch.FloatTensor([+1.4, -1.4, -0.7, 2.3, 0.3]).reshape(1, -1)

  assert torch.allclose(compute_loss(test_net, test_net(x), torch.LongTensor([4])),
                        torch.FloatTensor([7.486063259420916e-05]),
                        atol = 5e-7
                        )
  assert torch.allclose(compute_loss(test_net, test_net(x), torch.LongTensor([3])),
                        torch.FloatTensor([9.500075340270996]),
                        atol = 1e-3
                        )
