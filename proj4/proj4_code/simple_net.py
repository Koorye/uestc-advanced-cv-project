import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self):
        '''
        Init function to define the layers and loss function

        Note: Use 'sum' reduction in the loss_criterion. Read Pytorch documention
        to understand what it means
        '''
        super().__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3), padding=0),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3), padding=0),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(500, 100),
            nn.Linear(100, 15),
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        '''
        Perform the forward pass with the net

        Args:
        -   x: the input image [Dim: (N,C,H,W)]
        Returns:
        -   y: the output (raw scores) of the net [Dim: (N,15)]
        '''
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)
