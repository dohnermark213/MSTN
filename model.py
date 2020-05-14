import torch 
import torch.nn  as nn
from torch import Tensor

from torch import optim

class MSTN(nn.Module):
    def __init__(self, args, gen = None, dis = None, clf = None):
        super(MSTN, self).__init__()

        self.gen = gen
        self.dis = dis
        self.clf = clf  

        if self.gen == None :
            self.gen = Generator(args)
        if self.dis == None :
            self.dis = Discriminator(args)
        if self.clf == None :
            self.clf = Classifier(args)

        self.n_features = args.n_features
        self.n_class = args.n_class

