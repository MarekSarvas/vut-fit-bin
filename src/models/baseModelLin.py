# Author: Marek Sarvas
import torch
import torch.nn as nn

MAIN_MODEL_PATH = 'models/LinearNet/LinearNet_trained.pt'


class LinearNet(nn.Module):
    def __init__(self, num_classes=2):
        super(LinearNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64,  kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1*28*28, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        #x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = torch.softmax(x, dim=1)
        return x


def linearnet(device, pretrained=False, model_path=MAIN_MODEL_PATH, **kwargs):
    model = LinearNet(**kwargs)
    if pretrained:
        model_path = MAIN_MODEL_PATH if model_path is None else model_path
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    return model
