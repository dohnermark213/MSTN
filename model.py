'''
more formally, our total obejective can be written as follows:

    L(X_s, Y_s, X_t) = L_c(X_s, Y_s) + L_dc(X_s, X_t) + L_sm(X_s, Y_s, X_t)

cross entropy loss +  domain adversarial similarity loss + centroid alignment

'''

adversarial_loss = torch.nn.BCELoss()
classification_loss =  torch.nn.CrossEntropyLoss()
center_loss = torch.nn.MSELoss(reduction='sum')

def eval_batch(model, x_s, x_t, true_s, true_t, opt, train, args):

     s_clf, s_gen, s_dis = model(x_s)
     t_clf, t_gen, t_dis = model(x_t)

     #classification loss 

     #generator loss 

     #semantic loss

     loss = C+S*args.lam+G*args.lam

