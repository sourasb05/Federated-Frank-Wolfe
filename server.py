import torch 
import os
from client import *
from utils import *
import numpy as np


# implementation of FedAvg server


class FedAvg(Server):
    def __init__(self, device, dataset, algorithm, model, batch_size, step_size, glob_iters, local_iters, optimizer, num_users, times):
        super().__init__(device, dataset, algorithm, model[0], batch_size, step_size, glob_iters, local_iters, optimizer, num_users, times)

        # Initialize data for all users

        data = read_dataset(dataset)   # read_dataset will return 
        total_users = len(data[0])   
        for i in range(total_users):   # creating user instances for n number of users
            id, train, test = read_user_data(i, data, dataset)
            user = UserAvg(device, id, train, test, model, batch_size, step_size, local_iters, optimizer)   # Creating the instance of the users. 
            self.users.append(user)  # storing the user instances into a list 
            self.total_train_samples +=user.train_samples

            print("Number of Users / total users:"), i, "/" , total_users)
        print("FedAvg server creation Finished")
    
    def send_grads(self):
        assert (self.users is not None )