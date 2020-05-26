'''
cross entropy loss + domain adversarial similarity loss + centroid alignment

L(X_s, Y_s, X_t) = L_c(X_s, Y_s) + L_dc(X_s, X_t) + L_sm(X_s, Y_s, X_t)

'''

import torch
import torch.nn as nn
from torch import Tensor
from torch import optim

from modules import  Generator, Discriminator, Classifier

import numpy as np

class MSTN(nn.Module):
    def __init__(self, args):
        
        self.gen = Generator(args)
        self.dis = Discriminator(args)
        self.clf = Classifier(args)


        self.n_features = args.n_features
        self.n_class = args.n_class

        self.s_center = torch.zeros((args.n_class, args.n_features), requires_grad = False, device=args.device)
        self.t_center = torch.zeros((args.n_class, args.n_features), requires_grad = False, device=args.device)

        def forward(self, x):
            features = self.gen(x)
            C = self.clf(features)
            D = self.dis(features)
            return C, features, D

