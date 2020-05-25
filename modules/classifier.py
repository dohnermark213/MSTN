import  torch.nn as nn
from torch import tensor  

class  Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()

        self.main = nn.Sequential(
                nn.Linear(args.n_features, args.n_class),
                nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.main(x)
