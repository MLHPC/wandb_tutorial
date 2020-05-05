import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import wandb
import argparse

from net import Net

parser = argparse.ArgumentParser(
    description='Sample NMIST to demonstrate how to use W&B')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam'])

args = parser.parse_args()

# wandb setup
hyperparams = {
    'epochs': args.epochs,
    'batch_size': args.batch_size,
    'lr': args.lr
}
wandb.init(config=hyperparams, project='sample-pytorch-mnist',
           name='wandb-test-run')

# dataset preparation
normalization = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

dataset_train = torchvision.datasets.MNIST(
    root='./dataset', train=True, download=True, transform=normalization)
dataloader_train = torch.utils.data.DataLoader(
    dataset_train, batch_size=args.batch_size, shuffle=True)

dataset_test = torchvision.datasets.MNIST(
    root='./dataset', train=False, download=True, transform=normalization)
dataloader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=args.batch_size, shuffle=True)

classes = tuple(np.linspace(0, 9, 10, dtype=np.uint8))

# model setup
net = Net()
criterion = nn.CrossEntropyLoss()
if args.optimizer == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=args.lr)
else:
    optimizer = optim.Adam(lr=args.lr)

# training
for epoch in range(args.epochs):
    for i, (X, Y) in enumerate(dataloader_train, 0):
        optimizer.zero_grad()

        Yhat = net(X)
        loss = criterion(Yhat, Y)
        loss.backward()
        optimizer.step()

        wandb.log({'epoch': epoch+1, 'batch': i+1, 'loss': loss.item()})
        print('[{:d}, {:5d}] loss: {:.3f}'.format(
            epoch + 1, i + 1, loss.item()))


# testing
correct = 0
total = 0
with torch.no_grad():
    for (X, Y) in iter(dataloader_test):
        Yhat = net(X)
        _, predicted = torch.max(Yhat.data, 1)
        total += Y.size(0)
        correct += (predicted == Y).sum().item()
        loss = criterion(Yhat, Y).item()

test_accuracy = float(correct/total)
print('Accuracy: {:.2f} %%'.format(100 * test_accuracy))

wandb.run.summary["test_accuracy"] = test_accuracy
