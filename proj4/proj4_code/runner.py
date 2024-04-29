import os
import matplotlib.pyplot as plt
import torch.utils

from torch.autograd import Variable

from proj4_code.image_loader import ImageLoader
from proj4_code.dl_utils import predict_labels, compute_loss


class Trainer():
  '''
  This class makes training the model easier
  '''

  def __init__(self,
               data_dir,
               model,
               optimizer,
               model_dir,
               train_data_transforms,
               test_data_transforms,
               batch_size=100,
               load_from_disk=True,
               cuda=False
               ):
    self.model_dir = model_dir

    self.model = model

    self.cuda = cuda
    if cuda:
      self.model.cuda()

    dataloader_args = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    self.train_dataset = ImageLoader(
        data_dir, split='train', transform=train_data_transforms)
    self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,
                                                    **dataloader_args)

    self.test_dataset = ImageLoader(
        data_dir, split='test', transform=test_data_transforms)
    self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True,
                                                   **dataloader_args
                                                   )

    self.optimizer = optimizer

    self.train_loss_history = []
    self.validation_loss_history = []
    self.train_accuracy_history = []
    self.validation_accuracy_history = []

    # load the model from the disk if it exists
    if os.path.exists(model_dir) and load_from_disk:
      checkpoint = torch.load(os.path.join(self.model_dir, 'checkpoint.pt'))
      self.model.load_state_dict(checkpoint['model_state_dict'])
      self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    self.model.train()

  def save_model(self):
    '''
    Saves the model state and optimizer state on the dict
    '''
    torch.save({
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict()
    }, os.path.join(self.model_dir, 'checkpoint.pt'))

  def train(self, num_epochs):
    '''
    The main train loop
    '''
    self.model.train()
    for epoch_idx in range(num_epochs):
      for batch_idx, batch in enumerate(self.train_loader):
        if self.cuda:
          input_data, target_data = Variable(
              batch[0]).cuda(), Variable(batch[1]).cuda()
        else:
          input_data, target_data = Variable(batch[0]), Variable(batch[1])

        output_data = self.model(input_data)
        loss = compute_loss(self.model, output_data, target_data)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

      self.train_loss_history.append(float(loss))
      self.model.eval()
      self.eval_on_test()
      self.validation_accuracy_history.append(self.get_accuracy(split='test'))
      self.train_accuracy_history.append(self.get_accuracy(split='train'))
      self.model.train()

      if epoch_idx % 1 == 0:
        print('Epoch:{}, Loss:{:.4f}'.format(epoch_idx+1, float(loss)))
        # self.save_model()

  def eval_on_test(self):
    '''
    Get loss on test set
    '''
    test_loss = 0.0

    num_examples = 0
    for batch_idx, batch in enumerate(self.test_loader):
      if self.cuda:
        input_data, target_data = Variable(
            batch[0]).cuda(), Variable(batch[1]).cuda()
      else:
        input_data, target_data = Variable(batch[0]), Variable(batch[1])

      num_examples += input_data.shape[0]
      output_data = self.model.forward(input_data)
      loss = compute_loss(self.model,
                          output_data, target_data, is_normalize=False)

      test_loss += float(loss)

    self.validation_loss_history.append(test_loss/num_examples)

    return self.validation_loss_history[-1]

  def get_accuracy(self, split='test'):
    '''
    Get the accuracy on the test/train dataset
    '''
    self.model.eval()

    num_examples = 0
    num_correct = 0
    for batch_idx, batch in enumerate(self.test_loader if split is 'test' else self.train_loader):
      if self.cuda:
        input_data, target_data = Variable(
            batch[0]).cuda(), Variable(batch[1]).cuda()
      else:
        input_data, target_data = Variable(batch[0]), Variable(batch[1])

      num_examples += input_data.shape[0]
      predicted_labels = predict_labels(self.model, input_data)
      num_correct += torch.sum(predicted_labels == target_data).cpu().item()

    self.model.train()

    return float(num_correct)/float(num_examples)

  def plot_loss_history(self):
    '''
    Plots the loss history
    '''
    plt.figure()
    ep = range(len(self.train_loss_history))

    plt.plot(ep, self.train_loss_history, '-b', label = 'training')
    plt.plot(ep, self.validation_loss_history, '-r', label = 'validation')
    plt.title("Loss history")
    plt.legend()
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.show()

  def plot_accuracy(self):
    '''
    Plots the accuracy history
    '''
    plt.figure()
    ep = range(len(self.train_accuracy_history))
    plt.plot(ep, self.train_accuracy_history, '-b', label = 'training')
    plt.plot(ep, self.validation_accuracy_history, '-r', label = 'validation')
    plt.title("Accuracy history")
    plt.legend()
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.show()
