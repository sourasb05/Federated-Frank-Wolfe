from model import *
from server import Fed_Avg_Server
from client import Fed_Avg_Client
from utils import select_users, read_data, read_user_data #, RMSE_nuclear
import argparse
import torch.nn as nn 

def main(dataset, model, fl_algorithm, optimizer, fl_aggregator, step_size, glob_iters, local_iters, batch_size, times, gpu):
    exp_no=0

    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

    while exp_no < times:
        if model == "CNN":
            if dataset == "MNIST":
                model = cnn_Mnist().to(device), model
                loss = nn.NLLLoss()
                
            elif dataset == "FMNIST":
                model = cnn_Fmnist().to(device), model
                loss = nn.NLLLoss()

            elif dataset == "CIFAR10":
                model = cnn_Cifar10().to(device), model
                loss = nn.NLLLoss()
            
            elif dataset == "EMNIST":
                model = cnn_Emnist().to(device), model
                loss = nn.NLLLoss()
            
            elif dataset == "CELEBA":
                model = cnn_Celeba().to(device), model
                loss = nn.NLLLoss()

            elif dataset == "CIFAR100":
                model = cnn_Cifar100().to(device), model
                loss = nn.NLLLoss()
            
            elif dataset == "MOVIELENS_1m" and dataset == "MOVIELENS_100k":
                # model = MatrixFactorization(num_users, num_movies, embedding_size)
                print("model is in progress....")
            
        else:
            print(" do nothing")
        
        if fl_algorithm == "FedAvg":
            users = []   # the list of the object of the users
            data = read_data(dataset) 
            # print(data)
            input("interrupt from line 50 in main")

            # print("len_data",len(data[1]))
            # input("press")
            """
            data[0] : client id
            data[1] : train data
            data[2] : test data
            
            """
              # read_dataset will return 
            total_users = len(data[0])  
            server = Fed_Avg_Server(fl_aggregator, model)
            for i in range(0,total_users):
                train, test = read_user_data(i,data,dataset)
                # print("len(train)", len(train))
                # print(i)
                data_ratio = len(data[1])/len(train)
                # print("data_ratio",data_ratio) ## The ratio is not fixed yet
                user = Fed_Avg_Client(model, optimizer, loss, train, test, local_iters, step_size, batch_size, data_ratio, device)   # Creating the instance of the users. 
                users.append(user)
            # print(users)
            
        for i in range(0,glob_iters):
            print("----- Global iteration [",i,"]-----")
            server.send_parameters(users)   # server send parameters to every users
            server.evaluate(users)  # evaluate global model
            selected_users = select_users(users)
            #print("len of selected users",len(selected_users))
            for user in selected_users:
                user.local_train()

            server.global_update(users, selected_users)
            



        exp_no+=1 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="CIFAR10", choices=["MNIST", "FMNIST", "CIFAR10", "EMNIST", "CIFAR100", "SYNTHETIC", "MOVIELENS_1m", "MOVIELENS_100k"])
    parser.add_argument("--model", type=str, default="CNN")
    parser.add_argument("--times", type=int, default=1 )
    parser.add_argument("--fl_algorithm", type=str, default= "FedAvg")
    parser.add_argument("--optimizer", type=str, default="GD", choices=["GD", "SGD", "PGD", "PSGD"])
    parser.add_argument("--step_size", type=float, default=0.01)
    parser.add_argument("--glob_iters", type=int, default=10)
    parser.add_argument("--local_iters", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=124)
    parser.add_argument("--device", type=int, default=0, choices=[0,1,2,3,4,5,6,7,8] )
    parser.add_argument("--fl_aggregator", type=str, default="simple_averaging", choices = ["simple_averaging", "weighted_averaging"])    

    args = parser.parse_args()


    print("=" * 60)
    print("Summary of training process:")
    print("FL Algorithm: {}".format(args.fl_algorithm))
    print("model: {}".format(args.model))
    print("optimizer: {}".format(args.optimizer))
    print("Aggregator: {}".format(args.fl_aggregator))
    print("Step_size: {}".format(args.step_size))
    print("Batch size: {}".format(args.batch_size))
    print("Global_iters: {}".format(args.glob_iters))
    print("Local_iters: {}".format(args.local_iters))
    print("experiments: {}".format(args.times))
    print("device : {}".format(args.device))
    print("=" * 60)



    main(dataset=args.dataset,
        model=args.model,
        fl_algorithm=args.fl_algorithm,
        optimizer=args.optimizer,
        fl_aggregator = args.fl_aggregator,
        step_size=args.step_size,
        glob_iters=args.glob_iters,
        local_iters=args.local_iters,
        batch_size = args.batch_size,
        times=args.times,
        gpu = args.device)

    
        




