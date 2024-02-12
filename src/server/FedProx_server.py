import torch 
import os
import copy
import h5py
import numpy as np
import pandas as pd
from tqdm import trange
from datetime import date
import matplotlib.pyplot as plt
from src.client.FedProx_client import FedProx_Client
from src.utils.utils import select_users, read_data, read_user_data 
import numpy as np
import torch.nn.init as init

# implementation of FedAvg server


class FedProx_Server():
    def __init__(self,args, model, loss, device):

        """
        Global model
        """

        self.global_model = copy.deepcopy(model)
        self.global_model_name = args.model_name
        self.aggregator_name = args.fl_aggregator
        self.algorithm = args.fl_algorithm
        self.dataset_name= args.dataset
        """
        Iterations
        """
        self.global_iters = args.global_iters
        self.local_iters = args.local_iters
        
        self.batch_size = args.batch_size
        self.num_users_perGR = args.num_users_perGR
        

        """
        Hyperparameters
        """
        self.learning_rate = args.alpha
        self.lambda_prox = args.lambda_0
        self.num_labels=args.num_labels

      
        self.exp_no = args.exp_no
        # self.current_directory = current_directory
        self.device = device
        
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
            user = FedProx_Client(model,
                                loss,
                                total_users,
                                train,
                                test, 
                                self.local_iters, 
                                self.learning_rate,
                                self.lambda_prox, 
                                self.batch_size, 
                                data_ratio,
                                device)  
            self.users.append(user)


        
    def global_update(self, selected_users): 
        
        
        N = len(selected_users)
    
        for user in selected_users:
            for g_param, l_param in  zip(self.global_model.parameters(), user.local_model.parameters()):
                if self.aggregator_name == "simple_averaging":
                    g_param.data = g_param.data + (1/N) * l_param.data.clone()
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

    def initialize_parameters_to_zero(self):
        for param in self.global_model.parameters():
            if param.requires_grad:
                init.zeros_(param)

    def train(self):
        
        for t in trange(self.global_iters):
            selected_users = select_users(self.users, self.num_users_perGR)
            self.send_parameters(selected_users)   # server send parameters to every users
            
            print("number of selected users",len(selected_users))
            for user in selected_users:
                user.local_train(self.global_model)
            
            self.initialize_parameters_to_zero()  # Because we are averaging parameters
            self.global_update(selected_users)
            self.evaluate(selected_users)  # evaluate global model

        # self.save_file()
        #self.plot_result()       
                
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
            train_stats = user.global_eval_train_data(self.global_model)
            test_stats = user.global_eval_test_data(self.global_model)

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
        alg = str(self.exp_no) + "_dataset_" + str(self.dataset_name) + "_aggregator_" + str(self.aggregator_name) + "_fl_algorithm_" + str(self.algorithm) + \
            "_model_" + str(self.global_model_name) + "_" + d1
        
   
        print(alg)
       
        directory_name = self.algorithm + "/" + self.dataset_name + "/" + str(self.global_model_name) + "/perf/" +  str(self.num_labels)
        # Check if the directory already exists
        if not os.path.exists("./results/"+directory_name):
        # If the directory does not exist, create it
            os.makedirs('./results/' + directory_name)

        with h5py.File("./results/"+ directory_name + "/" + '{}.h5'.format(alg), 'w') as hf:
            hf.create_dataset('exp_no', data=self.exp_no)
            hf.create_dataset('lambda_prox', data=self.lambda_prox)
            hf.create_dataset('lr', data=self.learning_rate)
            hf.create_dataset('global_rounds', data=self.global_iters)
            
            hf.create_dataset('global_train_accuracy', data=self.avg_train_accuracy_list)
            hf.create_dataset('global_train_loss', data=self.avg_train_loss_list)
            hf.create_dataset('global_test_accuracy', data=self.avg_test_accuracy_list)
            hf.create_dataset('global_test_loss', data=self.avg_test_loss_list)

            hf.close()

    def plot_result(self):
        
        #print(self.avg_train_accuracy_list)

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
        # ax[1].set_xticks(range(0, self.global_iters, int(self.global_iters/5)))
        ax[1].legend(prop={"size":12})

        
        directory_name = self.algorithm + "/" + self.dataset_name + "/" + str(self.global_model_name) + "/" + "plot"
        # Check if the directory already exists
        if not os.path.exists("./results/"+directory_name):
        # If the directory does not exist, create it
            os.makedirs('./results/' + directory_name)

        plt.draw()
       
        plt.savefig("./results/"+ directory_name  + "/" + "_lr_" + str(self.learning_rate) + \
            "_lambda_prox_" + str(self.lambda_prox) + "_global_iters_" + str(self.global_iters) + '.png')
    
        # Show the graph
        plt.show()
