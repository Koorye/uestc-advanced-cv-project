import torch
import torch.nn as nn

from torchvision.models import alexnet


class MyAlexNet(nn.Module):
  def __init__(self):
    '''
    Init function to define the layers and loss function

    Note: Use 'sum' reduction in the loss_criterion. Ready Pytorch documention
    to understand what it means

    Note: Do not forget to freeze the layers of alexnet except the last one

    Download pretrained alexnet using pytorch's API (Hint: see the import
    statements)
    '''
    super(MyAlexNet, self).__init__()

    layers_list = list(alexnet(pretrained=True).children())
    self.cnn_layers = layers_list[0]
    for i in [0, 3, 6, 8, 10]:
        self.cnn_layers[i].weight.requires_grad = False
        self.cnn_layers[i].bias.requires_grad = False

    self.avgpool = layers_list[1]
    self.fc_layers = nn.Sequential(
        layers_list[2][0],
        layers_list[2][1],
        layers_list[2][2],
        layers_list[2][3],
        layers_list[2][4],
        layers_list[2][5],
        nn.Linear(4096, 15, bias=True),
    )

    for i in [1, 4]:
        self.fc_layers[i].weight.requires_grad = False
        self.fc_layers[i].bias.requires_grad = False
    self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')

  def forward(self, x: torch.tensor) -> torch.tensor:
    '''
    Perform the forward pass with the net

    Args:
    -   x: the input image [Dim: (N,C,H,W)]
    Returns:
    -   y: the output (raw scores) of the net [Dim: (N,15)]
    '''

    model_output = None
    model_output = x.repeat(1, 3, 1, 1) # as AlexNet accepts color images
    model_output = self.cnn_layers(model_output)
    model_output = self.avgpool(model_output)
    model_output = torch.flatten(model_output, 1)
    model_output = self.fc_layers(model_output)
    return model_output
