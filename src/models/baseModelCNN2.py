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


class Stage(nn.Module): 
   def __init__(self, K, in_channels, init_in_channels, out_channels, kernel_size, default_kernel_size, genotype):                         
       super(Stage, self).__init__()
       self.nodes = nn.ModuleList([])
       self.stage_genotype = genotype
       self.K = K                                                                                                                   
       self.def_input_node = nn.Conv2d(in_channels=init_in_channels, out_channels=out_channels, kernel_size=default_kernel_size, padding=2)                                                                                                     
       for _ in range(K):
           self.nodes.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1)) 
                                                                                                                               
       self.def_output_node = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1)
                                                                 
   def forward(self, x):                                         
       x = self.def_input_node(x)
       if not self.check_for_connection():
           print("HELP")
           return x
       tmp_outputs = []
       
       for _ in range(self.K):
           tmp_outputs.append(0)

       tmp_outputs[0] = self.nodes[0](x) 

       for node_idx, node_connection in enumerate(self.stage_genotype):
           for input_node_idx, bit in enumerate(node_connection):
               if bit == '1':
                   tmp_outputs[node_idx+1] += tmp_outputs[input_node_idx] 
                   self.nodes[input_node_idx].has_output_connection = True
                   self.nodes[input_node_idx].has_input_connection = True

           if not torch.is_tensor(tmp_outputs[node_idx+1]):
               tmp_outputs[node_idx+1] = x

           if self.nodes[node_idx+1].has_input_connection:
               tmp_outputs[node_idx+1] = self.nodes[node_idx+1](tmp_outputs[node_idx+1])

       for i in range(self.K-1):
           if not self.nodes[i].has_output_connection and self.nodes[i].has_input_connection:
               tmp_outputs[-1] += tmp_outputs[i] 

       x = self.def_output_node(tmp_outputs[-1])
       return x
 


def basenet(device, pretrained=False, model_path=MAIN_MODEL_PATH, **kwargs):
    model = BaseNet(**kwargs)
    if pretrained:
        model_path = MAIN_MODEL_PATH if model_path is None else model_path
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    return model
