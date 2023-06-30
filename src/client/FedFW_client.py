import copy
import math
import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader
from src.utils.oracles import LMO_l1, LMO_l2
from src.Optimizer import FW

class FedFW_Client():

    def __init__(self, 
                model, 
                optimizer_name, 
                loss, 
                n,  # number of participating clients
                train_set, 
                test_set, 
                local_iters, 
                eta_t,  # learning rate
                lambda_t,  # regularizer
                kappa, # Constraint 
                batch_size, 
                data_ratio, 
                device):
        
        self.x_it = copy.deepcopy(model[0])
        self.g_it = copy.deepcopy(model[0])
        self.x_bar_t = copy.deepcopy(model[0])
        self.s_it = copy.deepcopy(model[0])
        self.eval_model = copy.deepcopy(model[0])
        self.train_samples = len(train_set)
        self.test_samples = len(test_set)
        self.local_iters = local_iters
        self.eta_t = eta_t
        # print("eta_t :", self.eta_t)
        # input("press")
        self.lambda_t = lambda_t
        self.kappa = kappa
        self.batch_size = batch_size
        self.device = device
        self.optimizer_name = optimizer_name
        self.loss = loss
        self.data_ratio = data_ratio
        
        self.trainloader = DataLoader(train_set, self.batch_size)
        self.testloader =  DataLoader(test_set, self.batch_size)
        
        self.testloaderfull = DataLoader(test_set, self.test_samples)
        self.trainloaderfull = DataLoader(train_set, self.train_samples)
        self.iter_trainloader = iter(self.trainloader)   
        self.iter_testloader = iter(self.testloader)   
        self.iter_trainloaderfull = iter(self.trainloaderfull)   
        self.iter_testloaderfull = iter(self.testloaderfull)      
        
        self.participation_prob = 1.0
        self.n = n # number of participating clients
    
        self.optimizer = FW(self.x_it.parameters(), kappa=self.kappa)
        
     
    def selection(self):
        outcomes = [0,1]
        weights = [1-self.participation_prob, self.participation_prob]
        participation_choice = random.choices(outcomes, weights=weights)[0]

        return participation_choice
    
    def get_next_batch(self):
        try:
        # get a new batch
            # print("iter trainloader",self.iter_trainloader)
            (X,y) = next(self.iter_trainloaderfull)
            return (X.to(self.device), y.to(self.device))        

        except StopIteration:
             # restart the generator if the previous generator is exhausted.
            self.iter_trainloaderfull = iter(self.trainloaderfull)
            (X, y) = next(self.iter_trainloaderfull)
        return (X.to(self.device), y.to(self.device))
        
    def set_parameters(self, glob_model):
        for l_param, g_param in zip(self.x_it.parameters(), glob_model.parameters()):
            l_param.data = g_param.data

        
    def FW_LMO(self):
        all_param= [] 
        param_size = []
        for j, param in enumerate(self.g_it.parameters()):
          
            all_param.append(param.data.view(-1))
            param_size.append(len(param.data.view(-1)))
          
        param_concat = torch.cat(all_param, dim=0)
        lmo_s_it = LMO_l1(param_concat.cpu().numpy(), self.kappa)
        lmo_s_it = torch.from_numpy(lmo_s_it)
        # print(s_it)
        # input("press")
        param_proj = torch.split(lmo_s_it, param_size, dim=0)
        param_size_now = []
        #print(param_proj)
        #input("press")
        for proj , param in zip(param_proj, self.s_it.parameters()):
            parameter_size = param.shape
            param.data = proj.view(*parameter_size)
            # print(param.data)
        #print(self.s_it.parameters())
        #input("press")
            # print("After projection :", param.data)


    def local_train(self, x_bar_t):
        
        self.x_it.train()
        for iters in range(0, self.local_iters):
            
            X, y = self.get_next_batch()
            self.optimizer.zero_grad()
            output = self.x_it(X)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step(self.g_it, self.x_it, x_bar_t, self.n, self.lambda_t)
            self.FW_LMO()
            self.s_it.to(self.device)
            
            for x_it_param , s_it_param in zip(self.x_it.parameters(), self.s_it.parameters()):
                x_it_param.data = (1-self.eta_t)*x_it_param.data + self.eta_t*s_it_param.data

    def update_parameters(self, new_params):
        for param, new_param in zip(self.eval_model.parameters(), new_params):
            param.data = new_param.data.clone()

    def global_eval_train_data(self, global_update):
        self.eval_model.eval()
        train_acc = 0
        loss = 0
        self.update_parameters(global_update.parameters())
        for x, y in self.trainloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.eval_model(x)
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
            # avg_test_acc = test_acc / y.shape[0]
        return train_acc, y.shape[0], loss

    def global_eval_test_data(self, global_update):
        self.eval_model.eval()
        test_acc = 0
        loss = 0
        self.update_parameters(global_update.parameters())
        for x, y in self.testloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.eval_model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
            # avg_test_acc = test_acc / y.shape[0]
        return test_acc, y.shape[0], loss