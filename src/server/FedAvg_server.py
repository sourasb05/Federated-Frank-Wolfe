import torch 
import os
import copy
import h5py
import numpy as np
import pandas as pd
from tqdm import trange
from datetime import date
import matplotlib.pyplot as plt
from src.client.FedAvg_client import *
from src.utils.utils import *
import numpy as np


# implementation of FedAvg server


class Fed_Avg_Server():
    def __init__(self, aggregator_name, 
                 model,
                 global_iters):
        self.global_model = copy.deepcopy(model)
        self.aggregator_name = aggregator_name
        self.global_iters = global_iters
        
    def global_update(self, users, selected_users): 
        
        
        N = len(selected_users)

        # for param in self.global_model.parameters():
        #    param.data = torch.zeros_like(param.data)
        
        for user in selected_users:
            param_size_g_param = []
            param_size_l_param = []
            for g_param, l_param in  zip(self.global_model.parameters(), user.local_model.parameters()):
                param_size_g_param.append(len(g_param.view(-1)))
                param_size_l_param.append(len(l_param.view(-1)))
            
        for user in selected_users:
            for g_param, l_param in  zip(self.global_model.parameters(), user.local_model.parameters()):
                if self.aggregator_name == "simple_averaging":
                    # print("g_param:",g_param.data)
                    # print("l_param:",l_param.data)
                    # input("press")
                    g_param.data = g_param.data + (1/N) * l_param.data 
                elif self.aggregator_name == "weighted_averaging":
                    g_param.data = g_param.data + user.data_ratio * l_param.data
                else:
                    assert(f"Assertion Error: no aggregator found")
    
    def send_parameters(self, users):   
        if len(users) == 0:
            assert(f"AssertionError : The client can't be zero")
        else:
            for user in users:
                user.set_parameters(self.global_model)
    
    def train(self, users):
        
        for t in trange(self.global_iters):
            self.send_parameters(users)   # server send parameters to every users
            selected_users = select_users(users)
            print("number of selected users",len(selected_users))
            for user in selected_users:
                user.local_train()
            self.evaluate(users)  # evaluate global model
            # print(selected_users)
            # input("press")
            self.global_update(users, selected_users)

                
                
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
            train_stats = user.global_eval_train_data()
            test_stats = user.global_eval_test_data()

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

        print("Average train loss :", avg_train_loss.item())
        print("Average test loss :", avg_test_loss.item())
        print("Average train accuracy :", avg_train_accuracy)
        print("Average test accuracy :", avg_test_accuracy)
        
       
    def save_file(self):
        today = date.today()
        d1 = today.strftime("%d_%m_%Y")
       
        print("exp_no ", self.exp_no)
        alg = str(self.exp_no) + "_dataset_" + str(self.dataset) + "_aggregator_" + str(self.aggregator_name) + "_fl_algorithm_" + str(self.fl_algorithm) + \
            "_model_" + str(self.model_name) + "_" + d1
        
   
        print(alg)
       
        directory_name = self.fl_algorithm + "/" + self.dataset + "/" + str(self.model_name)
        # Check if the directory already exists
        if not os.path.exists("./results/"+directory_name):
        # If the directory does not exist, create it
            os.makedirs('./results/' + directory_name)

        with h5py.File("./results/"+ directory_name + "/" + '{}.h5'.format(alg), 'w') as hf:
            hf.create_dataset('exp_no', data=self.exp_no)
            hf.create_dataset('kappa', data=self.kappa) 
            hf.create_dataset('lambda_0', data=self.lambda_0)
            hf.create_dataset('eta_0', data=self.eta_t) 
            hf.create_dataset('eta_type', data=self.eta_type)
            hf.create_dataset('lambda_type', data=self.lambda_type)
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
        ax[1].set_ylabel("Loss")
        ax[1].set_xticks(range(0, self.global_iters, int(self.global_iters/5)))
        ax[1].legend(prop={"size":12})
        
        directory_name = self.fl_algorithm + "/" + self.dataset + "/" + str(self.model_name) + "/" + "plot"
        # Check if the directory already exists
        if not os.path.exists("./results/"+directory_name):
        # If the directory does not exist, create it
            os.makedirs('./results/' + directory_name)

        plt.draw()
       
        plt.savefig("./results/"+ directory_name  + "/" + "_lamdba_0" + str(self.lambda_0) + \
            "_kappa_" + str(self.kappa) + "_global_iters_" + str(self.global_iters) + '.png')

        # Show the graph
        plt.show()
