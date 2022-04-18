# Project: VUT FIT BIN - Neuroevolution
# Author: Marek Sarvas
# Login: xsarva00                   
# Date: 2021/2022                       
# Module: Function for training and evaluating NN -> used in fitness computation

import torch
import torchvision
import torchvision.datasets as datasets

from torch.utils.data import DataLoader
from torch import optim

from chromosome import Chromosome


def train(epochs, model, cuda, dataset="mnist"):
    # load data + create dataloader used in training for retrieving batches 
    if dataset == "fashion":
        train = datasets.FashionMNIST(root='../data', train=True, download=False, transform=torchvision.transforms.ToTensor())
    elif dataset == "cifar10":
        train = datasets.CIFAR10(root='../data', train=False, download=False, transform=torchvision.transforms.ToTensor())
    else:
        train = datasets.MNIST(root='../data', train=True, download=False, transform=torchvision.transforms.ToTensor())
    loader = DataLoader(train, batch_size=1024, shuffle=True, pin_memory=True)
    
    # set loss function and gradient descent optimizer
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
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
            optimizer.zero_grad()
            y = model(images)
            loss = criterion(y, labels)
            #print("LOSS: {}".format(loss)) 

            # train set accuracy

            # update weights of the model
            #loss.backward()
            loss.backward(retain_graph=True)
            optimizer.step()

            y = torch.argmax(y, dim=1)
            correct += torch.sum(y == labels)
            total += len(y)
        print('Epoch: {}, Accuracy: {:.5f}'.format(epoch, correct/total)) 
    return model


def eval(model, cuda, dataset="mnist"):
    # load data + create dataloader used in evaluating for retrieving batches 
    if dataset == "fashion":
        test = datasets.FashionMNIST(root='../data', train=False, download=False, transform=torchvision.transforms.ToTensor())
    elif dataset == "cifar10":
        test = datasets.CIFAR10(root='../data', train=False, download=False, transform=torchvision.transforms.ToTensor())
    else:
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
    cuda = True
    if torch.cuda.is_available() and cuda:
        device = torch.device('cuda')
    else:
        device  = torch.device('cpu')
    data="cifar10"
    net = Chromosome(stages=3, nodes=[3, 4,5], genotype=[['1', '00'], ['0', '01', '000'], ['0', '00', '000', '0000']], dataset=data)
    #net = Chromosome(stages=3, nodes=[3, 4, 5], genotype=[['1', '00'], ['0', '11', '010'], ['0', '01', '000', '0101']], dataset=data)
    #net = Chromosome(stages=2, nodes=[4, 5], genotype=[['1', '00', '000'], ['0', '11', '101', '0000']])
    #net = Chromosome(stages=3, nodes=[4, 5], genotype=[['1', '11', '110'], ['1', '10', '101', '1011']], dataset=data)
    if cuda:
        net.cuda()
    #for a in net.children():
        #print(a)
    trained = train(30, net, cuda, data)
    eval(trained, cuda, data)


