import copy
import math
import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader
from src.Optimizer import MySGD

class Fed_Avg_Client():

    def __init__(self,
                 args, 
                 model, 
                 loss, 
                 train_set, 
                 test_set, 
                 data_ratio, 
                 device):
        
        self.local_model = copy.deepcopy(model)
        self.eval_model = copy.deepcopy(model)
        self.train_samples = len(train_set)
        self.test_samples = len(test_set)
        self.device = device
        self.loss = loss
        self.data_ratio = data_ratio

        self.local_iters = args.local_iters
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.optimizer_name = args.optimizer
        self.kappa = args.kappa
        
        if self.optimizer_name == "GD" or self.optimizer_name == "PGD":
            self.trainloader = DataLoader(train_set, self.train_samples)
            self.testloader =  DataLoader(test_set, self.test_samples)
            self.tot_epoch = 1
            


        elif self.optimizer_name == "SGD" or self.optimizer_name == "PSGD":
            self.trainloader = DataLoader(train_set, self.batch_size)
            self.testloader =  DataLoader(test_set, self.batch_size)
            self.tot_epoch = math.ceil(torch.div(self.train_samples, self.batch_size))
        
        else:
            raise ValueError('No optimizer found')
        
        self.optimizer = MySGD(self.local_model.parameters(), lr=self.learning_rate)
        
       
        
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
        glob_model_param=glob_model.parameters()
        # Store the initial local model parameters
        if isinstance(glob_model_param, nn.Parameter):
            print("It is nn.Param")
            for l_param, g_param in zip(self.local_model.parameters(), glob_model_param):
                l_param.data = g_param.data.clone()
        elif isinstance(glob_model_param, list):
            print("It is a list")
            for idx, l_param in enumerate(self.local_model.parameters()):
                l_param.data = glob_model_param[idx].clone()

    def l2_projection(self, x, radius):
        """
        Projects the input vector x onto the L2 ball with the given radius.
        """
        
        l2_norm = torch.norm(x.float(), p=2)
        # print(l2_norm)
        # print(radius)
        # input("press")
        
        if l2_norm <= radius:
            return x
        else:
            return torch.mul(torch.div(x, l2_norm), radius)


    """def Projection_l1(x, alpha):
            shape = x.shape
            x = x.reshape(-1)
            proj = euclidean_proj_l1ball(x, alpha)
        return proj.reshape(*shape)
        
        def Projection_l2(x, alpha):
            return x / max(alpha, np.linalg.norm(x, 2))
"""

        
    def projection(self):
        all_param= [] 
        param_size = []
        for j, param in enumerate(self.local_model.parameters()):
          
            all_param.append(param.data.view(-1))
            param_size.append(len(param.data.view(-1)))
          
        param_concat = torch.cat(all_param, dim=0)
        l2_radius = self.kappa
        param_concat_proj = self.l2_projection(param_concat, l2_radius)
    
        
        param_proj = torch.split(param_concat_proj, param_size, dim=0)
        param_size_now = []
        # print(param_proj.shape())
        for proj , param in zip(param_proj, self.local_model.parameters()):
            # print(" i am here")
            parameter_size = param.shape
            param.data = proj.view(*parameter_size)
        
            # print("After projection :", param.data)

    def projection_2(self):
        shape = self.local_model.parameters().shape
        # for param in self.local_model.parameters():
        #    param.data 
        # print(shape)
        # input("press")
        parameters = self.local_model.parameters().reshape(-1)
        l2_radius=1
        param_proj = self.l2_projection(parameters, l2_radius)

        param_proj = param_proj.reshape(*shape)

        # self.local_model.parameters() = param_proj
              
    def update_eval_parameters(self, new_params):
        for param, new_param in zip(self.eval_model.parameters(), new_params):
            param.data = new_param.data.clone()


    def local_train(self):
        
        self.local_model.train()
        for epoch in range(0, self.tot_epoch):
            X, y = self.get_next_batch()
            self.optimizer.zero_grad()
            output = self.local_model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()
        if self.optimizer_name == "PGD" or self.optimizer_name == "PSGD":
            self.projection()


    def global_eval_train_data(self, global_model):
        self.eval_model.eval()
        train_correct = 0
        loss = 0
        self.update_eval_parameters(global_model.parameters())
        for x, y in self.trainloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.local_model(x)
            train_correct += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
        return train_correct, y.shape[0], loss

    def global_eval_test_data(self, global_model):
        self.eval_model.eval()
        test_correct = 0
        loss = 0
        self.update_eval_parameters(global_model.parameters())
        for x, y in self.testloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.local_model(x)
            test_correct += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
        return test_correct, y.shape[0], loss