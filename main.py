import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision
import os
import PIL.Image as pil_image
import numpy as np
from senet import ressenet50
import argparse
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 SENet')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0
class RandomRotate(object):
    def __init__(self, max_ang):
        assert max_ang > 0
        self.max_ang = max_ang
    def __call__(self, x, mode="reflect"):
        angle = np.random.randint(-self.max_ang, self.max_ang)
        x = x.rotate(angle)
        return x
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    RandomRotate(10),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])
trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
model = ressenet50().cuda()

criterion = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
scheduler = StepLR(optimizer, step_size=30, gamma=0.5)


def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (X_batch, y_batch) in enumerate(trainloader):

        X_batch = Variable(X_batch).cuda()
        y_batch = Variable(y_batch).cuda()
        y_preds = model(X_batch)
        optimizer.zero_grad()
        loss = criterion(y_preds, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]
        _, predicted = torch.max(y_preds.data, 1)
        total += y_batch.size(0)
        correct += predicted.eq(y_batch.data).cpu().sum()

        print (batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (X_batch, y_batch) in enumerate(testloader):
        X_batch = Variable(X_batch).cuda()
        y_batch = Variable(y_batch).cuda()
        y_preds = model(X_batch)
        loss = criterion(y_preds, y_batch)

        test_loss += loss.data[0]
        _, predicted = torch.max(y_preds.data, 1)
        total += y_batch.size(0)
        correct += predicted.eq(y_batch.data).cpu().sum()

        print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': model._modules if use_cuda else model,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/model.pth')
        best_acc = acc
if __name__ == "__main__":
    for epoch in xrange(200):
        train(epoch)
        test(epoch)
        scheduler.step(epoch)
