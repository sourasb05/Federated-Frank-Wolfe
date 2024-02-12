import copy
import math
import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader
#from src.utils.oracles import LMO_l1, LMO_l2
from src.Optimizer import PerturbedSGD

class FedDR_Client():
# n =  number of participating clients
    def __init__(self, args, model, loss, n, train_set, test_set, data_ratio, device):
        
        self.x_bar_k = copy.deepcopy(model)  # global model
        self.y_ik = copy.deepcopy(model)
        self.x_hat = copy.deepcopy(model)
        self.x_hat_prev = copy.deepcopy(model)
        self.x_ik = copy.deepcopy(model) # local model
        self.delta_x_hat_k = copy.deepcopy(model)
        
        self.eval_model = copy.deepcopy(model) # evaluate global model
        
        self.train_samples = len(train_set)
        self.test_samples = len(test_set)
        self.local_iters = args.local_iters
        self.batch_size = args.batch_size
        self.device = device
        self.loss = loss
        self.data_ratio = data_ratio
        self.eta = args.eta_0
        self.lr = args.lr
        self.alpha = args.alpha
        
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
        self.optimizer = PerturbedSGD(self.x_ik.parameters(), self.lr, self.alpha)
     
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
        for x_bar_k_param, g_param in zip(self.x_bar_k.parameters(), global_model.parameters()):
            x_bar_k_param.data = g_param.data

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

        # update y_ik

        for y_ik_param, x_bar_k_param, x_ik_param in zip(self.y_ik.parameters(),self.x_bar_k.parameters(),self.x_ik.parameters()):
            y_ik_param.data = y_ik_param.data + self.alpha*(x_bar_k_param.data - x_ik_param.data)
        
        
        # self.optimizer.update_v_star(self.y_ik)
     
        for iters in range(0, self.local_iters):
            
            self.x_ik.train()
            X, y = self.get_next_batch()
            self.optimizer.zero_grad()
            output = self.x_ik(X)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step(self.y_ik)
        
        #update x_hat
        for x_hat_param, x_ik_param, y_ik_param in zip(self.x_hat.parameters(), self.x_ik.parameters(),
                                                        self.y_ik.parameters()):
            x_hat_param.data = 2*x_ik_param.data - y_ik_param.data
            
        # update delta x_hat_k (Algorithm 1 line 7)
            
        for delta_x_hat_k_param, x_hat_param, x_hat_prev_param in zip(self.delta_x_hat_k.parameters(), 
                                                                      self.x_hat.parameters(),
                                                                      self.x_hat_prev.parameters()):
            delta_x_hat_k_param.data = x_hat_param.data - x_hat_prev_param.data
            x_hat_prev_param.data = x_hat_param.data.clone()


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