import copy
import os
import h5py
import numpy as np
import pandas as pd
from tqdm import trange
from datetime import date
import matplotlib.pyplot as plt
from src.client.FedFW_client import FedFW_Client
from src.utils.utils import read_data, read_user_data, select_users
import torch.nn.init as init
import torch

class FedFW_Server():
    def __init__(self, args, model, loss, device):
        self.x_bar_t = copy.deepcopy(model)
        self.s_bar_t = copy.deepcopy(model)
        self.loaded_global_model = copy.deepcopy(model)
        self.aggregator_name = args.fl_aggregator
        self.fl_algorithm = args.fl_algorithm
        self.global_iters = args.global_iters
        self.model_name = args.model_name
        self.dataset = args.dataset
        self.eta_type = args.eta_type
        self.lambda_type = args.lambda_type
        self.kappa = args.kappa
        self.num_users_perGR = args.num_users_perGR
        self.exp_no = args.exp_no
        self.num_labels = args.num_labels
        self.fw_gaps = 0.0

        self.avg_train_loss_list = []
        self.avg_test_loss_list = []
        self.avg_train_accuracy_list = []
        self.avg_test_accuracy_list = []
        self.FW_gap = []

        if self.eta_type == "constant_eta":
           self. eta_t = args.eta_0 /  (args.global_iters ** (2/3))
        else:
            self.eta_t = 1
        if args.lambda_type == "constant_lambda":
            self.lambda_t = args.lambda_0 * ( args.global_iters ** (1/3)) 
        else:
            self.lambda_t = args.lambda_0
                
        data = read_data(args)
        total_users = len(data[0])
        self.users = []
        for i in range(0,total_users):
            train, test = read_user_data(i,data,args.dataset)
            data_ratio = len(data[1])/len(train)
            user = FedFW_Client(args, 
                                model, 
                                loss, 
                                train, 
                                test,
                                self.eta_t,
                                self.lambda_t,
                                data_ratio,
                                device)   # Creating the instance of the users. 
                    
            self.users.append(user)

    def save_model(self, glob_iter, model_name):
        if model_name == "step_direction":
            model_path = "./models/step_direction/"
            print(model_path)
            # input("press")
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            print(f"saving global model at round {glob_iter}")
            torch.save(self.s_bar_t, os.path.join(model_path, "step_direction_" + str(glob_iter)  + ".pt"))

    def load_model(self):
        model_path = "./models/step_direction/step_direction_99.pt"
        
        assert (os.path.exists(model_path))
        self.loaded_global_model = torch.load(model_path)

        for p in self.loaded_global_model.parameters():
            print(p.grad.data)
            input("press")
            # print(p.grad.data)
        

    def model_exists(self):
        return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))

        
    def global_update(self,selected_users, t): 
        
        for user in selected_users:
            for x_bar_t_param, x_it_param in zip(self.x_bar_t.parameters(), user.x_it.parameters()):
                x_bar_t_param.data += (1/len(selected_users))*x_it_param.data.clone()
                
    def s_bar_t_update(self,selected_users, t): 
        
        for user in selected_users:
            for s_bar_t_param, s_it_param in zip(self.s_bar_t.parameters(), user.s_it.parameters()):
                s_bar_t_param.data += (1/len(selected_users))*s_it_param.data.clone()
        
        """for s in self.s_bar_t.parameters():
            print(s.data)
            input("press")"""
    def send_parameters(self, users):   
        if len(users) == 0:
            assert(f"AssertionError : The client can't be zero")
        else:
            for user in users:
                user.set_parameters(self.x_bar_t)


    def initialize_parameters_to_zero(self):
        for param in self.sum_sit.parameters():
            if param.requires_grad:
                init.zeros_(param)
    def initialize_global_parameters_to_zero(self):
        for param in self.x_bar_t.parameters():
            if param.requires_grad:
                init.zeros_(param)
    
    def initialize_s_bar_t_to_zero(self):
        for param in self.s_bar_t.parameters():
            if param.requires_grad:
                init.zeros_(param)

    



    def train(self):
          
        for t in trange(self.global_iters):
            
            selected_users = select_users(self.users, self.num_users_perGR)
            print("number of selected users",len(selected_users))
            self.send_parameters(selected_users)   # server send parameters to every users
            for user in selected_users:
                user.local_train(self.x_bar_t)
            
            self.initialize_global_parameters_to_zero()
            self.initialize_s_bar_t_to_zero()
            self.s_bar_t_update(selected_users, t)
            self.global_update(selected_users, t)
            self.evaluate_FW_gap(selected_users, t)
            self.evaluate(self.users)  # evaluate global model
            # self.save_model(t, "step_direction")


    def evaluate_FW_gap(self, users, t):
        self.fw_gaps = 0.0
        for user in users:
            self.fw_gaps += user.fw_gap*(1/len(users))
        print(f"Frank wolfe Gaps at global round {t} :", self.fw_gaps.item())
        
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
            train_stats = user.global_eval_train_data(self.x_bar_t)
            test_stats = user.global_eval_test_data(self.x_bar_t)

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
        alg = str(self.exp_no) + "_dataset_" + str(self.dataset) + "_aggregator_" + str(self.aggregator_name) + "_fl_algorithm_" + str(self.fl_algorithm) + \
            "_model_" + str(self.model_name) + "_lamdba_0_" + str(self.lambda_t) + "_eta_0_" + str(self.eta_t) + \
            "_kappa_" + str(self.kappa) + "_global_iters_" + str(self.global_iters) + "_" + d1
        
   
        print(alg)
       
        directory_name = self.fl_algorithm + "/" + self.dataset + "/" + str(self.model_name) + "/" + str(self.eta_type) + "/" + str(self.lambda_type) + "/hyperparameters/" + str(self.num_labels)
        # Check if the directory already exists
        if not os.path.exists("./results/"+directory_name):
        # If the directory does not exist, create it
            os.makedirs('./results/' + directory_name)

        with h5py.File("./results/"+ directory_name + "/" + '{}.h5'.format(alg), 'w') as hf:
            hf.create_dataset('exp_no', data=self.exp_no)
            hf.create_dataset('kappa', data=self.kappa) 
            hf.create_dataset('lambda_0', data=self.lambda_t)
            hf.create_dataset('eta_0', data=self.eta_t) 
            hf.create_dataset('eta_type', data=self.eta_type)
            hf.create_dataset('lambda_type', data=self.lambda_type)
            hf.create_dataset('global_rounds', data=self.global_iters)
            
            hf.create_dataset('global_train_accuracy', data=self.avg_train_accuracy_list)
            hf.create_dataset('global_train_loss', data=self.avg_train_loss_list)
            hf.create_dataset('global_test_accuracy', data=self.avg_test_accuracy_list)
            hf.create_dataset('global_test_loss', data=self.avg_test_loss_list)
            hf.create_dataset('fw_gap',data=self.fw_gaps)

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
       
        plt.savefig("./results/"+ directory_name  + "/" + "_lamdba_0" + str(self.lambda_t) + "_eta_0_" + str(self.eta_t) + \
            "_kappa_" + str(self.kappa) + "_global_iters_" + str(self.global_iters) + '.png')

        # Show the graph
        plt.show()
