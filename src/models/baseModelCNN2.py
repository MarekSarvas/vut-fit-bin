# Author: Marek Sarvas
import torch
import torch.nn as nn

MAIN_MODEL_PATH = 'models/BaseNet/BaseNet_trained.pt'


class BaseNet(nn.Module):
    def __init__(self, num_classes=2):
        super(BaseNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2),

            #nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            #nn.Dropout(),
            #nn.Linear(32*7*7, 256),
            #nn.ReLU(inplace=True),
            #nn.Dropout(),
            #nn.Linear(512, 512),
            #nn.ReLU(inplace=True),
            #nn.Linear(256, num_classes),
            nn.Linear(32*8*8, num_classes),
        )
        self.a = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
                                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
                                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2),
                                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2)])

    def forward(self, x):
        for node in range(2):
            x = self.a[node](x)
            x = self.features(x)

        #x = self.features(x)
        #print('X: ', x.shape)
        #x = self.conv2d_3(x)
        #print('X conv: ', x.shape)
        x = torch.flatten(x, 1)
        #print('X flat: ', x.shape)
        x = self.classifier(x)
        x = torch.softmax(x, dim=1)
        return x


def basenet(device, pretrained=False, model_path=MAIN_MODEL_PATH, **kwargs):
    model = BaseNet(**kwargs)
    if pretrained:
        model_path = MAIN_MODEL_PATH if model_path is None else model_path
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    return model
