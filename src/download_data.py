import torchvision.datasets as datasets
import torchvision

train = datasets.MNIST(root='../data', train=True, download=True, transform=None)
test = datasets.MNIST(root='../data', train=False, download=True, transform=torchvision.transforms.ToTensor())

print('MNIST downloaded.')
print('Train data: ', len(train))
print('Test data: ', len(test))

