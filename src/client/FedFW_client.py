import copy
import math
import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader
from src.utils.oracles import LMO_l1, LMO_l2
from src.Optimizer import FedFW
import torch.nn.init as init

class FedFW_Client():

    def __init__(self,
                args, 
                model, 
                loss,
                train_set, 
                test_set, 
                eta_t,
                lambda_0,  
                data_ratio, 
                device):
        """
        Models
        """
        self.x_it = copy.deepcopy(model)  # local model
        self.s_it = copy.deepcopy(model) # step direction
        self.y_it = copy.deepcopy(model)
        self.eval_model = copy.deepcopy(model) # evaluate global model
        
        for param in self.y_it.parameters():
            if param.requires_grad:
                init.zeros_(param)
        """
        Hyperparameters
        """
        self.lambda_0 = lambda_0
        self.kappa = args.kappa
        self.batch_size = args.batch_size
        self.optimizer_name = args.optimizer
        self.batch_size = args.batch_size
        self.all_batch = args.all_batch
        self.local_iters = args.local_iters
        self.algorithm = args.fl_algorithm
        self.device = device
        self.loss = loss
        self.data_ratio = data_ratio


        """
        Train and test sets
        """
        self.train_samples = len(train_set)
        self.test_samples = len(test_set)
        
        if self.all_batch == 0:
            self.trainloader = DataLoader(train_set, self.train_samples)
            self.testloader =  DataLoader(test_set, self.test_samples)
            self.iter_trainloader = iter(self.trainloader)   
            self.iter_testloader = iter(self.testloader) 
            self.epochs = 1  
        else:
            self.trainloader = DataLoader(train_set, self.batch_size)
            self.testloader =  DataLoader(test_set, self.batch_size)
            self.iter_trainloader = iter(self.trainloader)   
            self.iter_testloader = iter(self.testloader)
            self.epochs = math.ceil(torch.div(self.train_samples, self.batch_size))
            

        self.trainloaderfull = DataLoader(train_set, self.train_samples)
        self.testloaderfull = DataLoader(test_set, self.test_samples)
        self.iter_trainloaderfull = iter(self.trainloaderfull)   
        self.iter_testloaderfull = iter(self.testloaderfull)      
        
        
        

        
        
        # self.participation_prob = 1.0
        # self.n = n # number of participating clients
    
        # self.optimizer = FW(self.x_it.parameters(), eta_t=self.eta_t, kappa=self.kappa, device=self.device)
        self.optimizer = FedFW(self.x_it.parameters(),
                                lambda_0=self.lambda_0,
                                eta_t=eta_t,
                                eta_type=args.eta_type,
                                lambda_type=args.lambda_type,
                                num_client_iter=self.local_iters,
                                step_direction_func=LMO_l2,
                                kappa=args.kappa,
                                algorithm=args.fl_algorithm)
     
    
    
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
        for x_it_param, g_param in zip(self.x_it.parameters(), glob_model.parameters()):
            x_it_param.data = g_param.data.clone()

    
    def get_step_directions(self):
        state_dict = self.optimizer.state_dict()
        return [state_dict['state'][key]["step_direction"] for key in state_dict['state'].keys()]

    def get_eta(self):
        state_dict = self.optimizer.state_dict()
        eta_t_list = [state_dict['state'][key]["eta_t"] for key in state_dict['state'].keys()]
        return eta_t_list[0]

    def initialize_s_it_to_zero(self):
        for param in self.s_it.parameters():
            if param.requires_grad:
                init.zeros_(param)
    
    
    def local_train(self, x_bar_t):
        self.x_it.train()
        # self.initialize_s_it_to_zero()
        for iters in range(0, self.local_iters):
            for epoch in range(self.epochs):
                X, y = self.get_next_batch()
                self.optimizer.zero_grad()
                output = self.x_it(X)
                loss = self.loss(output, y)
            
                loss.backward(retain_graph=True)
                __, list1 = self.optimizer.step(self.s_it, x_bar_t, self.y_it, self.algorithm)
        # print(list1)    
        # input("press")
        for original_param, new_param in zip(self.s_it.parameters(), list1):
            original_param.grad = new_param.data.clone()

         
    def update_eval_parameters(self, new_params):
        for param, new_param in zip(self.eval_model.parameters(), new_params):
            param.data = new_param.data.clone()

    def global_eval_train_data(self, global_update):
        self.eval_model.eval()
        train_acc = 0
        loss = 0
        self.update_eval_parameters(global_update.parameters())
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
        self.update_eval_parameters(global_update.parameters())
        for x, y in self.testloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.eval_model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
            # avg_test_acc = test_acc / y.shape[0]
        return test_acc, y.shape[0], loss