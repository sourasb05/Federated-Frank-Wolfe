import copy
import math
import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader
from src.utils.oracles import LMO_l1, LMO_l2
from src.Optimizer import FedFW

class FedFW_Client():

    def __init__(self, 
                model, 
                optimizer_name,
                loss,
                n,  # number of participating clients
                train_set, 
                test_set, 
                local_iters, 
                lambda_0,  # regularizer
                kappa, # Constraint 
                batch_size, 
                data_ratio, 
                device):
        
        self.x_it = copy.deepcopy(model)  # local model
        self.x_bar_t = copy.deepcopy(model) # global_model
        
        self.eval_model = copy.deepcopy(model) # evaluate global model
        self.model_bar = copy.deepcopy(list(self.x_it.parameters()))
        self.train_samples = len(train_set)
        self.test_samples = len(test_set)
        self.local_iters = local_iters
        self.lambda_0 = lambda_0
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
    
        # self.optimizer = FW(self.x_it.parameters(), eta_t=self.eta_t, kappa=self.kappa, device=self.device)
        self.optimizer = FedFW(self.x_it.parameters(),
                                server_model=self.x_bar_t,
                                lambda_0=self.lambda_0,
                                num_client_iter=local_iters,
                                step_direction_func=LMO_l2,
                                alpha=100)
     
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
        for x_bar_t_param, g_param in zip(self.x_bar_t.parameters(), glob_model.parameters()):
            x_bar_t_param.data = g_param.data

    def update_local_parameters(self):
        for param, new_param in zip(self.x_it.parameters(), self.model_bar):
            param.data = new_param.clone()

    def compare_networks(self):
        params1 = list(self.x_it.parameters())
        params2 = list(self.x_bar_t.parameters())
        if len(params1) != len(params2):
            print("len(params1) != len(params2)")
        else:
            print("lenth ok")
        for p1, p2 in zip(params1, params2):
            if p1.shape != p2.shape:
                print("p1.shape != p2.shape")
            else:
                print("shape ok")    

    def set_zero(self):
        for param in self.x_it.parameters():
            param.data.zero_()

    def set_server_parameters(self, model):
        for old_param, new_param in zip(self.model.parameters(), model.parameters()):
            old_param.data = new_param.data.clone()
        self.optimizer.set_server_model(model)

    def get_step_directions(self):
        state_dict = self.optimizer.state_dict()
        return [state_dict['state'][key]["step_direction"] for key in state_dict['state'].keys()]

    def get_eta(self):
        state_dict = self.optimizer.state_dict()
        eta_t_list = [state_dict['state'][key]["eta_t"] for key in state_dict['state'].keys()]
        return eta_t_list[0]


    def local_train(self):
     
     for iters in range(0, self.local_iters):
            
            self.x_it.train()
            X, y = self.get_next_batch()
            self.optimizer.zero_grad()
            output = self.x_it(X)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()
            # self.update_local_parameters()
    
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