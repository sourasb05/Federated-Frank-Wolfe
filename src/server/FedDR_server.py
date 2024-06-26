import os
import torch
import torch.optim
import torch.nn as nn
import copy
from tqdm import tqdm, trange
import h5py
import numpy as np
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
from src.utils.utils import select_users
import torch.nn.init as init
from src.client.FedDR_client import FedDR_Client
from src.utils.utils import select_users, read_data, read_user_data #, RMSE_nuclear

class FedDR_Server:

    def __init__(self, args, model, loss, device):
        
        self.x_bar_k = copy.deepcopy(model)
        
        self.fl_algorithm = args.fl_algorithm
        self.global_iters = args.global_iters
        self.model_name = args.model_name
        self.dataset = args.dataset
        self.eta = args.eta_0
        self.alpha = args.alpha
        self.exp_no = args.exp_no
        self.kappa = args.kappa
        self.num_labels = args.num_labels
        self.p = args.p
        self.avg_train_loss_list = []
        self.avg_test_loss_list = []
        self.avg_train_accuracy_list = []
        self.avg_test_accuracy_list = []
        self.users = []
        data = read_data(args)
        total_users = len(data[0])

        for i in range(0,total_users):
            train, test = read_user_data(i,data,args.dataset)
            # print("len(train)", len(train))
            # print(i)
            data_ratio = len(data[1])/len(train)
            # print("data_ratio",data_ratio) ## The ratio is not fixed yet
            # Creating the instance of the users. 
            user = FedDR_Client(args, model, loss, total_users, train, test, data_ratio, device) 
                                  
            self.users.append(user)

        
    def server_aggregation(self, selected_users):
        for user in selected_users:
            for x_bar_k_param, delta_x_hat_k_param in zip (self.x_bar_k.parameters(), user.delta_x_hat_k.parameters()):
                x_bar_k_param.data = x_bar_k_param.data + (1/len(selected_users))*delta_x_hat_k_param.data
    def l2_projection(self, x):
        """
        Projects the input vector x onto the L2 ball with the given radius.
        """
        
        l2_norm = torch.norm(x.float(), self.p)
        
        if l2_norm <= self.kappa:
            return x
        else:
            return torch.mul(torch.div(x, l2_norm), self.kappa)


    
    def projection(self):
        all_param= [] 
        param_size = []
        for j, param in enumerate(self.x_bar_k.parameters()):
          
            all_param.append(param.data.view(-1))
            param_size.append(len(param.data.view(-1)))
          
        param_concat = torch.cat(all_param, dim=0)
        param_concat_proj = self.l2_projection(param_concat)
    
        
        param_proj = torch.split(param_concat_proj, param_size, dim=0)
        param_size_now = []
        # print(param_proj.shape())
        for proj , param in zip(param_proj, self.x_bar_k.parameters()):
            # print(" i am here")
            parameter_size = param.shape
            param.data = proj.view(*parameter_size)
        
    def global_update(self,selected_users): 
        self.server_aggregation(selected_users)
        self.projection()
        # for x_bar_k_param in self.x_bar_k.parameters():
        #    x_bar_k_param.data = x_bar_k_param.data - self.eta*x_bar_k_param.grad.data
        
                
    def send_parameters(self, users):   
        if len(users) == 0:
            assert(f"AssertionError : The client can't be zero")
        else:
            for user in users:
                user.set_parameters(self.x_bar_k)


    def initialize_parameters_to_zero(self):
        for param in self.x_bar_k.parameters():
            if param.requires_grad:
                init.zeros_(param)
            

    def train(self):
       
        self.initialize_parameters_to_zero()  # initialize all parameters to zero
           
        
        for t in trange(self.global_iters):
            self.send_parameters(self.users)   # server send parameters to every users
            
            # selected_users = select_users(self.users)
            # print("number of selected users",len(selected_users))
            # for user in selected_users:
            for user in self.users:
                user.local_train()

            self.global_update(self.users)
            self.evaluate(self.users)  # evaluate global model
                
    def evaluate(self, users):
        tot_train_loss = 0
        tot_test_loss = 0
        tot_train_corrects = 0
        tot_test_corrects = 0
        tot_samples_train = 0
        tot_samples_test = 0
        
        """
            train_stats[0] = training accuracy
            train_stats[1] = number of training sample y.shape[0]
            train_stats[2] = training loss
            

            test_stats[0] = test accuracy
            test_stats[1] = y.shape[0]
            test_stats[2] = test loss
        """

        for user in users:
            train_stats = user.global_eval_train_data(self.x_bar_k)
            test_stats = user.global_eval_test_data(self.x_bar_k)

            tot_train_corrects += train_stats[0]
            tot_samples_train += train_stats[1]
            tot_train_loss += train_stats[2]

            
            tot_test_corrects += test_stats[0]
            tot_samples_test += test_stats[1]
            tot_test_loss += test_stats[2]


        avg_train_loss = tot_train_loss/len(users)
        avg_test_loss = tot_test_loss/len(users)
        avg_train_accuracy = tot_train_corrects/tot_samples_train
        avg_test_accuracy = tot_test_corrects/tot_samples_test

        self.avg_train_loss_list.append(avg_train_loss.item())
        self.avg_test_loss_list.append(avg_test_loss.item())
        self.avg_train_accuracy_list.append(avg_train_accuracy)
        self.avg_test_accuracy_list.append(avg_test_accuracy)

        print("Average train loss :", avg_train_loss.item())
        print("Average test loss :", avg_test_loss.item())
        print("Average train accuracy :", avg_train_accuracy)
        print("Average test accuracy :", avg_test_accuracy)


    
    def save_file(self):
        today = date.today()
        d1 = today.strftime("%d_%m_%Y")
       
        print("exp_no ", self.exp_no)
        alg = str(self.exp_no) + "_dataset_" + str(self.dataset) + "_fl_algorithm_" + str(self.fl_algorithm) + \
            "_model_" + str(self.model_name) + "_eta_" + str(self.eta) + \
            "_kappa_" + str(self.kappa) + "_global_iters_" + str(self.global_iters) + "_" + d1
        
   
        print(alg)
       
        directory_name = self.fl_algorithm + "/" + self.dataset + "/" + str(self.model_name) + "/perf/new/norm_" + str(self.p) + "/" + str(self.num_labels)
        # Check if the directory already exists
        if not os.path.exists("./results/"+directory_name):
        # If the directory does not exist, create it
            os.makedirs('./results/' + directory_name)

        with h5py.File("./results/"+ directory_name + "/" + '{}.h5'.format(alg), 'w') as hf:
            hf.create_dataset('exp_no', data=self.exp_no)
            hf.create_dataset('kappa', data=self.kappa) 
            hf.create_dataset('alpha', data=self.alpha)
            hf.create_dataset('eta', data=self.eta)
            hf.create_dataset('num_labels', data=self.num_labels)
            hf.create_dataset('global_rounds', data=self.global_iters)
            hf.create_dataset('global_train_accuracy', data=self.avg_train_accuracy_list)
            hf.create_dataset('global_train_loss', data=self.avg_train_loss_list)
            hf.create_dataset('global_test_accuracy', data=self.avg_test_accuracy_list)
            hf.create_dataset('global_test_loss', data=self.avg_test_loss_list)

            hf.close()

    def plot_result(self):
        
        print(self.avg_train_accuracy_list)

        fig, ax = plt.subplots(1,2, figsize=(12,6))

        ax[0].plot(self.avg_train_accuracy_list, label= "Train_accuracy")
        ax[0].plot(self.avg_test_accuracy_list, label= "Test_accuracy")
        ax[0].set_xlabel("Global Iteration")
        ax[0].set_ylabel("accuracy")
        ax[0].set_xticks(range(0, self.global_iters, int(self.global_iters/5)))
        ax[0].legend(prop={"size":12})
        ax[1].plot(self.avg_train_loss_list, label= "Train_loss")
        ax[1].plot(self.avg_test_loss_list, label= "Test_loss")
        ax[1].set_xlabel("Global Iteration")
        ax[1].set_xscale('log')
        ax[1].set_ylabel("Loss")
        ax[1].set_yscale('log')
        #ax[1].set_xticks(range(0, self.global_iters, int(self.global_iters/5)))
        ax[1].legend(prop={"size":12})
        
        directory_name = self.fl_algorithm + "/" + self.dataset + "/" + str(self.model_name) + "/" + "plot"
        # Check if the directory already exists
        if not os.path.exists("./results/"+directory_name):
        # If the directory does not exist, create it
            os.makedirs('./results/' + directory_name)

        plt.draw()
       
        plt.savefig("./results/"+ directory_name  + "/" + "_eta_" + str(self.eta) + \
            "_kappa_" + str(self.kappa) + "_global_iters_" + str(self.global_iters) + '.png')

        # Show the graph
        plt.show()
