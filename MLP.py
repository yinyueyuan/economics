import torch
from d2l import torch as d2l
from torch import nn
w=1
b=1
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
def train_concise(wd):
    net=nn.Sequential(nn.Linear(num_inputs,1))
    for param in net.parameters():
        param.data.normal_()
    loss=nn.MSELoss(reduction='none')
    num_epochs,lr=100,0.001
    trainer=torch.optim.SGD([{'params':net[0].weight,'weight_decay':wd},{'params':net[0].bias}],lr=lr)
    for epoch in range(num_epochs):
        for X,y in train_iter:
            trainer.zero_grad()
            l=loss(net(X),y)
            l.mean().backward()
            trainer.step()
        
