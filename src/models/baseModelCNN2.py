# Author: Marek Sarvas
import torch
import torch.nn as nn

MAIN_MODEL_PATH = 'models/BaseNet/BaseNet2_trained.pt'


class BaseNet2(nn.Module):
    def __init__(self, num_classes=2):
        super(BaseNet2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64*6*6, num_classes),
        )
       

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = torch.softmax(x, dim=1)
        return x


def basenet2(device, pretrained=False, model_path=MAIN_MODEL_PATH, **kwargs):
    model = BaseNet2(**kwargs)
    if pretrained:
        model_path = MAIN_MODEL_PATH if model_path is None else model_path
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    return model
