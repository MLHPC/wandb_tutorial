import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import wandb

from net import Net

EPOCHS = 2
BATCH_SIZE = 256
LR = 0.01

# wandb setup
wandb.init(project='sample-pytorch-mnist', name='wandb-test-run')
wandb.config.update({
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'lr': LR
})

# dataset preparation
normalization = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

dataset_train = torchvision.datasets.MNIST(
    root='./dataset', train=True, download=True, transform=normalization)
dataloader_train = torch.utils.data.DataLoader(
    dataset_train, batch_size=BATCH_SIZE, shuffle=True)

dataset_test = torchvision.datasets.MNIST(
    root='./dataset', train=False, download=True, transform=normalization)
dataloader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=BATCH_SIZE, shuffle=True)

classes = tuple(np.linspace(0, 9, 10, dtype=np.uint8))

# model setup
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR)

# training
for epoch in range(EPOCHS):
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
