import copy
import math
import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader
from src.Optimizer import MySGD

class FedProx_Client():

    def __init__(self, 
                 model, 
                 loss, 
                 total_users,
                 train_set, 
                 test_set, 
                 local_iters, 
                 learning_rate,
                 lambda_prox, 
                 batch_size, 
                 data_ratio, 
                 device):
        
        self.local_model = copy.deepcopy(model)
        self.train_samples = len(train_set)
        self.test_samples = len(test_set)
        self.local_iters = local_iters
        self.learning_rate = learning_rate
        self.lambda_prox = lambda_prox
        self.batch_size = batch_size
        self.device = device
        self.loss = loss
        self.data_ratio = data_ratio
        
        """
        Optimizer
        """
        
        self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.learning_rate)

        """
        1. testset
        2. trainset
        """
        if  self.local_iters == 1:
            self.trainloader = DataLoader(train_set, self.train_samples)
            self.testloader =  DataLoader(test_set, self.test_samples)
            self.iter_trainloader = iter(self.trainloader)   
            self.iter_testloader = iter(self.testloader) 
             
        else:
            self.trainloader = DataLoader(train_set, self.batch_size)
            self.testloader =  DataLoader(test_set, self.batch_size)
            self.iter_trainloader = iter(self.trainloader)   
            self.iter_testloader = iter(self.testloader)
            self.local_iters = math.ceil(torch.div(self.train_samples, self.batch_size))
            

        self.trainloaderfull = DataLoader(train_set, self.train_samples)
        self.testloaderfull = DataLoader(test_set, self.test_samples)
        self.iter_trainloaderfull = iter(self.trainloaderfull)   
        self.iter_testloaderfull = iter(self.testloaderfull)    

        self.testloaderfull = DataLoader(test_set, self.test_samples)
        self.trainloaderfull = DataLoader(train_set, self.train_samples)
        
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
        
    def set_parameters(self, global_model):
        for l_param, g_param in zip(self.local_model.parameters(), global_model.parameters()):
                l_param.data = g_param.data.clone()
        

    def local_train(self, global_model):
        
        self.local_model.train()
        for iters in range(0, self.local_iters):
            X, y = self.get_next_batch()
            self.optimizer.zero_grad()
            output = self.local_model(X)
            loss = self.loss(output, y)
            proximal_term = 0.0
            for param, g_param in zip(self.local_model.parameters(), global_model.parameters()):
                proximal_term += (self.lambda_prox / 2) * torch.norm(param - g_param) ** 2
            loss += proximal_term
            loss.backward()
            self.optimizer.step()

    def update_parameters(self, global_model):
           for l_param, g_param in zip(self.local_model.parameters(), global_model.parameters()):
                l_param.data = g_param.data.clone()


    def global_eval_train_data(self, global_model):
        self.local_model.eval()
        train_correct = 0
        loss = 0
        self.update_parameters(global_model)
        for x, y in self.trainloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.local_model(x)
            train_correct += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
            # avg_test_acc = test_acc / y.shape[0]
        return train_correct, y.shape[0], loss

    def global_eval_test_data(self, global_model):
        self.local_model.eval()
        test_correct = 0
        loss = 0
        self.update_parameters(global_model)
        for x, y in self.testloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.local_model(x)
            test_correct += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)

        return test_correct, y.shape[0], loss