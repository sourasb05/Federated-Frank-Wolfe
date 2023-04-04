import torch 
import os
import copy
from client import *
from utils import *
import numpy as np


# implementation of FedAvg server


class Fed_Avg_Server():
    def __init__(self, aggregator_name, model):
        self.global_model = copy.deepcopy(model[0])
        self.aggregator_name = aggregator_name
        
    def global_update(self, users, selected_users): 
        
        
        N = len(selected_users)

        for param in self.global_model.parameters():
            param.data = torch.zeros_like(param.data)
        for user in selected_users:
            for g_param, l_param in  zip(self.global_model.parameters(), user.local_model.parameters()):
                if self.aggregator_name == "simple_averaging":
                    g_param.data = g_param.data + (1/N)*l_param.data 
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
        
       
        