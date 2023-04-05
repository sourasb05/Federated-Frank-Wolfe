import copy
import math
import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader

class Fed_Avg_Client():

    def __init__(self, model, optimizer_name, loss, train_set, test_set, local_iters, learning_rate, batch_size, data_ratio, device):
        
        self.local_model = copy.deepcopy(model[0])
        self.train_samples = len(train_set)
        self.test_samples = len(test_set)
        self.local_iters = local_iters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device
        self.optimizer_name = optimizer_name
        self.loss = loss
        self.data_ratio = data_ratio
        
        if optimizer_name == "GD" or optimizer_name == "PGD":
            self.trainloader = DataLoader(train_set, self.train_samples)
            self.testloader =  DataLoader(test_set, self.test_samples)
            self.tot_epoch = 1
            self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.learning_rate)


        elif optimizer_name == "SGD" or optimizer_name == "PSGD":
            self.trainloader = DataLoader(train_set, self.batch_size)
            self.testloader =  DataLoader(test_set, self.batch_size)
            self.tot_epoch = math.ceil(torch.div(self.train_samples, self.batch_size))
            self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.learning_rate)
        
       
        else:
            print("no optimizer found")
        """
        1. evaluate testset
        2. evaluate trainset
        """
        self.testloaderfull = DataLoader(test_set, self.test_samples)
        self.trainloaderfull = DataLoader(train_set, self.train_samples)
        self.iter_trainloader = iter(self.trainloader)   
        self.iter_testloader = iter(self.testloader)      
        
        self.participation_prob = 1.0
    
    
     
    def selection(self):
        outcomes = [0,1]
        weights = [1-self.participation_prob, self.participation_prob]
        participation_choice = random.choices(outcomes, weights=weights)[0]

        return participation_choice
    
    def get_next_batch(self):
        try:
        # get a new batch
            # print("iter trainloader",self.iter_trainloader)
            (X,y) = next(self.iter_trainloader)
            return (X.to(self.device), y.to(self.device))        

        except StopIteration:
             # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.trainloader)
            (X, y) = next(self.iter_trainloader)
        return (X.to(self.device), y.to(self.device))
        
    def set_parameters(self, glob_model):
        for l_param, g_param in zip(self.local_model.parameters(), glob_model.parameters()):
            l_param.data = g_param.data

    def l2_projection(self, x, radius):
        """
        Projects the input vector x onto the L2 ball with the given radius.
        """
        
        l2_norm = torch.norm(x.float(), p=2)
        
        if l2_norm <= radius:
            return x
        else:
            return torch.mul(torch.div(x, l2_norm), radius)


        
    def projection(self):
        all_param= [] 
        param_size = []
        for j, param in enumerate(self.local_model.parameters()):
            all_param.append(param.view(-1))
            param_size.append(len(param.view(-1)))
        print("param_size :",param_size)
        param_concat = torch.cat(all_param, dim=0)
        l2_radius = 1
        param_concat_proj = self.l2_projection(param_concat, l2_radius)
    
        
        param_proj = torch.split(param_concat_proj, param_size, dim=0)
        param_size_now = []
        # print(param_proj.shape())
        for proj , param in zip(param_proj, self.local_model.parameters()):
            param.data = proj
            param_size_now.append(len(param.view(-1)))
        print("After projection param size :", param_size_now)


    def local_train(self):
        
        self.local_model.train()
        for iters in range(0, self.local_iters):
            
            for epoch in range(0, self.tot_epoch):
                X, y = self.get_next_batch()
                self.optimizer.zero_grad()
                output = self.local_model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
        if self.optimizer_name == "PGD":
            self.projection()


    def global_eval_train_data(self):
        self.local_model.eval()
        train_acc = 0
        loss = 0
        for x, y in self.trainloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.local_model(x)
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
            # avg_test_acc = test_acc / y.shape[0]
        return train_acc, y.shape[0], loss

    def global_eval_test_data(self):
        self.local_model.eval()
        test_acc = 0
        loss = 0
        for x, y in self.testloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.local_model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
            # avg_test_acc = test_acc / y.shape[0]
        return test_acc, y.shape[0], loss