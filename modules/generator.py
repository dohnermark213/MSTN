import torch.nn as nn
from torchvision.models import AlexNet, alexnet

import torch.utils.model_zoo as model_zoo
from torch.autograd import Function
from torch import tensor

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()

        self.input_size  = args.input_size

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            )
        self.avg = nn.AdaptiveAvgPool2d((6, 6))

        self.batch_norm = nn.BatchNorm2d(3)
        self.clf = nn.Sequential(
                nn.Linear(self.input_size, 1024),
            nn.ReLU(inplace= True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace= True),
            nn.Linear(1024, args.n_features),
        )
    
    def forward(self, x):
        x = self.batch_norm(x)
        x = x.view(x.size(0), -1)
        x = self.clf(x)
        return x 
       
