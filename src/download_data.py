import torchvision.datasets as datasets
import torchvision
import argparse





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="path to folder for storing datasets")

    params = parser.parse_args()

    train = datasets.MNIST(root=params.data_path, train=True, download=True, transform=None)
    test = datasets.MNIST(root=params.data_path, train=False, download=True, transform=torchvision.transforms.ToTensor())
    print('MNIST downloaded.')
    print('Train data: ', len(train))
    print('Test data: ', len(test))
    
    train = datasets.FashionMNIST(root=params.data_path, train=True, download=True, transform=None)
    test = datasets.FashionMNIST(root=params.data_path, train=False, download=True, transform=torchvision.transforms.ToTensor())
    print('FashionMNIST downloaded.')
    print('Train data: ', len(train))
    print('Test data: ', len(test))
    
    train = datasets.CIFAR10(root=params.data_path, train=True, download=True, transform=None)
    test = datasets.CIFAR10(root=params.data_path, train=False, download=True, transform=torchvision.transforms.ToTensor())
    print('CIFAR10 downloaded.')
    print('Train data: ', len(train))
    print('Test data: ', len(test))

