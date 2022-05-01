# Author: Marek Sarvas
# Login: xsarva00
# Script for training and evaluating baseline models.

import torch
import torchvision
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch import optim

from models.baseModelCNN import basenet
from models.baseModelCNN2 import basenet2
from train_eval_ea import train, eval


def train_base(epochs, model, cuda):
    # load data + create dataloader used in training for retrieving batches 
    train = datasets.MNIST(root='../data', train=True, download=True, transform=torchvision.transforms.ToTensor())
    loader = DataLoader(train, batch_size=2048, shuffle=True)
    
    # set loss function and gradient descent optimizer
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    correct = 0
    total = 0
    
    print("Training...") 
    for epoch in range(epochs):
        for batch, (images, labels) in enumerate(loader):
            if cuda:
                images = images.to('cuda')
                labels = labels.to('cuda')
            # prediction and loss
            y = model(images)
            loss = criterion(y, labels)
            
            # train set accuracy
            y = torch.argmax(y, dim=1)
            correct += torch.sum(y == labels)
            total += len(y)

            # update weights of the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch: {}, Accuracy: {:.5f}'.format(epoch, correct/total)) 
    return model


def eval_base(model, cuda):
    # load data + create dataloader used in evaluating for retrieving batches 
    test = datasets.MNIST(root='../data', train=False, download=True, transform=torchvision.transforms.ToTensor())
    loader = DataLoader(test, batch_size=128, shuffle=True)
  
    model.eval()
    
    correct = 0
    total = 0
    
    print("Evaluating on test set...") 
    for batch, (images, labels) in enumerate(loader):
        if cuda:
            images = images.to('cuda')
            labels = labels.to('cuda')            

        # model predictions
        y = model(images)
        
        # compute accuracy
        y = torch.argmax(y, dim=1)
        correct += torch.sum(y == labels)
        total += len(y)

    print('Model accuracy: {:.5f}'.format(correct/total)) 


if __name__ == '__main__':
    cuda = True
    if torch.cuda.is_available() and cuda:
        device = torch.device('cuda')
    else:
        device  = torch.device('cpu')
    net1 = basenet(device, pretrained=False, num_classes=10)
    net2 = basenet2(device, pretrained=False, num_classes=10)
    if cuda:
        net1.cuda()
        net2.cuda()
    dataset = "mnist"
    trained = train(20, net1, cuda, dataset)
    eval(trained, cuda, dataset)
    #trained = train(20, net2, cuda, dataset)
    #eval(trained, cuda, dataset)


