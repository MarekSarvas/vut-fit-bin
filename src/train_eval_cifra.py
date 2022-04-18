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
from models.baseModelLin import linearnet
from chromosome import Chromosome


def train(epochs, model, cuda):
    # load data + create dataloader used in training for retrieving batches 
    train = datasets.MNIST(root='../data', train=True, download=False, transform=torchvision.transforms.ToTensor())
    loader = DataLoader(train, batch_size=1024, shuffle=False, pin_memory=True)
    
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
            # print("LOSS: {}".format(loss)) 

            # train set accuracy
            y = torch.argmax(y, dim=1)
            correct += torch.sum(y == labels)
            total += len(y)

            # update weights of the model
            optimizer.zero_grad()
            #loss.backward()
            loss.backward(retain_graph=True)
            optimizer.step()

        print('Epoch: {}, Accuracy: {:.5f}'.format(epoch, correct/total)) 
    return model


def eval(model, cuda):
    # load data + create dataloader used in evaluating for retrieving batches 
    test = datasets.MNIST(root='../data', train=False, download=False, transform=torchvision.transforms.ToTensor())
    loader = DataLoader(test, batch_size=1024, shuffle=True, pin_memory=True)
  
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
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
    return correct/total


if __name__ == '__main__':
    cuda = False 
    if torch.cuda.is_available() and cuda:
        device = torch.device('cuda')
    else:
        device  = torch.device('cpu')
    net = Chromosome(stages=3, nodes=[3, 4,5], genotype=[['1', '01'], ['0', '01', '100'], ['0', '11', '101', '0001']])
    #net = Chromosome(stages=3, nodes=[3, 4, 5], genotype=[['0', '00'], ['0', '00', '000'], ['0', '00', '000', '0000']])
    if cuda:
        net.cuda()
    for a in net.children():
        print(a)
    trained = train(30, net, cuda)
    #eval(trained, cuda)


