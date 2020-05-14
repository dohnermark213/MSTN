import torch.nn as nn
from torch import tensor 

class Discriminator(nn.Module):
    def ___init__(self, args):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
        )
        
    def forward(self, x):
        return self.main(x)

class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()

        self.main = nn.Sequential()

    def forward(self, x):
        return self.main(x)

