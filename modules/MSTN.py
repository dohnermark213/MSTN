'''
cross entropy loss + domain adversarial similarity loss + centroid alignment

L(X_s, Y_s, X_t) = L_c(X_s, Y_s) + L_dc(X_s, X_t) + L_sm(X_s, Y_s, X_t)

'''

import torch
import torch.nn as nn
from torch import Tensor
from torch import optim

#from modules import  Generator, Discriminator, Classifier

from .generator import Generator
from .discriminator import Discriminator
from .classifier import Classifier

import numpy as np

class MSTN(nn.Module):
    def __init__(self, args, gen  = None, dis = None, clf = None):
        super(MSTN,self).__init__()

        self.gen =gen
        self.dis = dis
        self.clf = clf  

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


def update_centers(model, s_gen, t_gen, s_true, t_clf, args):
        source = torch.argmax(s_true, 1).reshape(t_clf.size(0),1).detach()
        target = torch.argmax(t_clf, 1).reshape(t_clf.size(0), 1).detach()

        s_center = torch.zeros(model.n_class, model.n_features, device=args.device)
        t_center = torch.zeros(model.n_class, model.n_features, device=args.device)

        s_zeros = torch.zeros(source.size()[1:], device = args.device)
        t_zeros = torch.zeros(target.size()[1:], device = args.device)

        for i in range(model.n_class):
            s_cur  = torch.where(source.eq(i), s_gen, s_zeros).mean(0)
            t_cur = torch.where(target.eq(i), t_gen, t_zeros).mean(0)
            s_center[i] = s_cur * (1-model.disc)+model.s_center[i]*model.disc
            t_center[i] = t_cur * (1-model.disc)+model.t_center[i]*model.disc

            return s_center,  t_center

adversarial_loss = torch.nn.BCELoss()
classification_loss = torch.nn.CrossEntropyLoss()
center_loss = torch.nn.MSELoss(reduction='sum')

def adaptation_factor(qq):
    return 2/(1+np.exp(-10*qq))-1

def one_hot(batch, classes):
    ones = torch.eye(classes)
    return  ones.index_select(0,  batch)

def accuracy(pred, true):
    pred = pred.argmax(1)
    return  (pred == true).float().mean()

def eval_batch(model, sx, tx, s_true, t_true, opt, train, args):
    if  train:
        opt.zero_grad()

    s_clf, s_gen, s_dis = model(sx)
    t_clf, t_gen, t_dis =  model(tx)
    
    #classification loss
    C_loss = classification_loss(s_clf, s_true.to(args.device))

    #generator  loss
    source_tag = torch.ones((sx.size(0), 1), device=args.device)
    target_tag =  torch.zeros((tx.size(0), 1), device=args.device)
    s_true_hot = one_hot(s_true, model.n_class)
    s_G_loss = adversarial_loss(s_dis, source_tag)
    t_G_loss = adversarial_loss(t_dis, target_tag)
    G_loss = (s_G_loss + t_G_loss)
    
    #semantic  loss  
    s_c, t_c  = update_centers(model, s_gen, t_gen, s_true_hot, t_clf, args)
    S_loss  =  center_loss(t_c, s_c)
    model.s_center = s_c.detach()
    model.t_center = t_c.detach()

    loss = C_loss + S_loss * args.lam + G_loss*args.lam
    if  train:
        loss.backward()
        opt.step()

    s_acc = accuracy(s_clf,  s_true.to(args.device))
    if t_true is not  None:
        t_acc = accuracy(t_clf, t_true)
    else:
        t_acc = torch.tensor(0)

    return  np.array([S_loss.item(), C_loss.item(), G_loss.item(), s_acc.item(), t_acc.item()])


def  run_epoch(model, opt, dataset, train, args):
    loss = np.zeros(5)
    device = args.device 
    if  train:
        model.train()
    else:
        model.eval()

    for sx, sy, tx, ty  in tqdm(dataset):
        loss+=eval_batch(model, sx.to(device), tx.to(device), sy, ty.to(device), opt, train, args)

    return  loss  


def fit(args, epochs,  model, opt, trainset, validset):
    out = list()
    for epoch in range(epochs):
        args.lam = adaptation_factor(epoch*1.0/epochs)
        train_loss = run_epoch(model, opt,  trainset,  train=True, args = args)
        valid_loss = run_epoch(model, None, validset, train=False, args=args)

        out.append((train_loss, valid_loss))

        if args.save_step:
            file = open(args.save+"_loss","wb")
            torch.save(model.state_dict(), args.save+'step')
            np.save(file, out)

    return out 

