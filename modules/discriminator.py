import torch.nn as nn
from torch import tensor  

'''for discriminator, we use the same architecture with  RevGrad,
 x-> 1024-> 1024-> 1
dropout is  used
'''

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
                nn.Linear(args.n_features, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x) 

