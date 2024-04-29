'''
Utilities to be used along with the deep model
'''

import torch


def predict_labels(model: torch.nn.Module, x: torch.tensor) -> torch.tensor:
    '''
    Perform the forward pass and extract the labels from the model output

    Args:
    -   model: a model (which inherits from nn.Module)
    -   x: the input image [Dim: (N,C,H,W)]
    Returns:
    -   predicted_labels: the output labels [Dim: (N,)]
    '''
    logits = model(x)
    _, predicted_labels = torch.max(logits, 1)
    return predicted_labels


def compute_loss(model: torch.nn.Module,
                 model_output: torch.tensor,
                 target_labels: torch.tensor,
                 is_normalize: bool = True) -> torch.tensor:
    '''
    Computes the loss between the model output and the target labels

    Args:
    -   model: a model (which inherits from nn.Module)
    -   model_output: the raw scores output by the net
    -   target_labels: the ground truth class labels
    -   is_normalize: bool flag indicating that loss should be divided by the batch size
    Returns:
    -   the loss value
    '''
    # compute loss, do not use torch.nn
    # loss = torch.nn.functional.cross_entropy(model_output, target_labels, reduction='sum')
    targets = torch.nn.functional.one_hot(target_labels, num_classes=model_output.size(1))
    loss = - torch.sum(targets * torch.log_softmax(model_output, dim=1), dim=1).sum()

    if is_normalize:
        loss /= model_output.size(0)
    
    return loss
