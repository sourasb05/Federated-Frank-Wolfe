from model import *
from src.server.FedAvg_server import Fed_Avg_Server
from src.server.FedFW_server import FedFW_Server
from src.client.FedAvg_client import Fed_Avg_Client
from src.client.FedFW_client import FedFW_Client
from src.utils.utils import select_users, read_data, read_user_data #, RMSE_nuclear
from src.MatrixComp import My_matrix_comp
import argparse
import torch.nn as nn 

def main(dataset, problem_type, model_name, fl_algorithm, optimizer, fl_aggregator, eta_0,lambda_0, eta_type,
        lambda_type, kappa, global_iters, local_iters, batch_size, times, gpu, num_users, num_labels):
    exp_no=0

    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")
    if problem_type == "MATRIX_COMP": problem_category = 1
    elif problem_type == "QAP": problem_category = 2
    else: problem_category = 3    
    
    while exp_no < times:

        if problem_type == "MATRIX_COMP":
        
            if  dataset == "MOVIELENS_100k" or dataset == "MOVIELENS_1m":
                data = read_data(dataset)
                # print(data)
                r = My_matrix_comp.matrix_comp(data)
                # input("press")
            
            print("code building in progress....")
    
        else:
            
            if model_name == "CNN":
                
                if dataset == "MNIST":
                    model = cnn_Mnist().to(device)
                    loss = nn.NLLLoss()
                        
                elif dataset == "FMNIST":
                    model = cnn_Fmnist(10).to(device)
                    loss = nn.CrossEntropyLoss()

                elif dataset == "CIFAR10":
                    model = cnn_Cifar10().to(device)
                    loss = nn.NLLLoss()
                    
                    
                elif dataset == "EMNIST":
                    model = cnn_Emnist().to(device)
                    loss = nn.NLLLoss()
                    
                elif dataset == "CELEBA":
                    model = cnn_Celeba().to(device)
                    loss = nn.NLLLoss()

                elif dataset == "CIFAR100":
                    model = cnn_Cifar100().to(device)
                    loss = nn.CrossEntropyLoss()

                        
            elif model_name == "MCLR":

                if(dataset == "human_activity"):
                    model = Mclr_Logistic(561,6).to(device)
                    loss = nn.NLLLoss()
                elif(dataset == "gleam"):
                    model = Mclr_Logistic(561,6).to(device)
                    loss = nn.NLLLoss()
                elif(dataset == "vehicle_sensor"):
                    model = Mclr_Logistic(100,2).to(device)
                    loss = nn.NLLLoss()
                elif(dataset == "SYNTHETIC"):
                    model = Mclr_Logistic(60,10).to(device)
                    loss = nn.NLLLoss()
                elif(dataset == "EMNIST"):
                    model = Mclr_Logistic(784,62).to(device)
                    loss = nn.NLLLoss()
                elif(dataset == "FMNIST"):
                    model = Mclr_Logistic(784,10).to(device)
                    loss = nn.NLLLoss() 

                elif(dataset == "CIFAR10"):
                    model = Mclr_Logistic(3072,10).to(device)
                    loss = nn.NLLLoss() 
                
                elif(dataset == "CIFAR100"):
                    model = Mclr_Logistic(3072,100).to(device)
                    loss = nn.NLLLoss() 

                else:#(dataset == "Mnist"):
                    model = Mclr_Logistic().to(device)
                    loss = nn.CrossEntropyLoss()
                    # loss = nn.NLLLoss()
                    print(model)
            
            elif(model_name == "DNN"):
                if(dataset == "human_activity"):
                    model = DNN(561,100,12).to(device)
                    loss = nn.NLLLoss()
                elif(dataset == "gleam"):
                    model = DNN(561,20,6).to(device)
                    loss = nn.NLLLoss()
                elif(dataset == "vehicle_sensor"):
                    model = DNN(100,20,2).to(device)
                    loss = nn.NLLLoss()
                elif(dataset == "SYNTHETIC"):
                    model = DNN(60,20,10).to(device)
                    loss = nn.NLLLoss()
                elif(dataset == "EMNIST"):
                    model = DNN(784,200,62).to(device)
                    loss = nn.NLLLoss()
                else:#(dataset == "Mnist"):
                    model = DNN2().to(device)
                    loss = nn.NLLLoss()
        

            
            
            else:
                print(" No problem selected")
           
        

        if problem_category == 3:
            users = []
            print(dataset)
            data = read_data(args)
            total_users = len(data[0])  
            if fl_algorithm == "FedFW":
                if eta_type == "constant_eta":
                    eta_t = eta_0 /  (global_iters ** (2/3))
                else:
                    eta_t = 1
                if lambda_type == "constant_lambda":
                    lambda_t = lambda_0 * ( global_iters ** (1/3)) 
                else:
                    lambda_t = lambda_0

                server = FedFW_Server(fl_aggregator, 
                                      fl_algorithm, 
                                      model, 
                                      model_name, 
                                      dataset, 
                                      global_iters, 
                                      lambda_t, 
                                      eta_t, 
                                      eta_type,
                                      lambda_type,
                                      kappa, 
                                      exp_no )
                for i in range(0,total_users):
                    train, test = read_user_data(i,data,dataset)
                    data_ratio = len(data[1])/len(train)
                    user = FedFW_Client(model, 
                                        optimizer, 
                                        loss, 
                                        total_users, 
                                        train, 
                                        test, 
                                        local_iters, 
                                        eta_t, 
                                        lambda_t, 
                                        eta_type,
                                        lambda_type,
                                        kappa, 
                                        batch_size, 
                                        data_ratio, 
                                        device)   # Creating the instance of the users. 
                    
                    users.append(user)
                
            elif fl_algorithm == "FedAvg":
                server = Fed_Avg_Server(fl_aggregator, 
                                        model,
                                        global_iters)
                
                for i in range(0,total_users):
                    train, test = read_user_data(i,data,dataset)
                    # print("len(train)", len(train))
                    # print(i)
                    data_ratio = len(data[1])/len(train)
                    # print("data_ratio",data_ratio) ## The ratio is not fixed yet
                    user = Fed_Avg_Client(model,
                                        optimizer, 
                                        loss, 
                                        train, 
                                        test, 
                                        local_iters, 
                                        eta_0,
                                        batch_size, 
                                        data_ratio, 
                                        device)   # Creating the instance of the users. 
                    users.append(user)

                
            server.train(users)
            #server.save_file()
            server.plot_result()
                
            """for i in range(0,glob_iters):
                print("----- Global iteration [",i,"]-----")
                server.send_parameters(users)   # server send parameters to every users
                server.evaluate(users)  # evaluate global model
                selected_users = select_users(users)
                #print("len of selected users",len(selected_users))
                for user in selected_users:
                    user.local_train()

                server.global_update(users, selected_users)"""
                



        exp_no+=1 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "FMNIST", "CIFAR10", "EMNIST", "CIFAR100", "CELEBA", "SYNTHETIC", "MOVIELENS_1m", "MOVIELENS_100k"])
    parser.add_argument("--problem_type", type=str, default="NN", choices=["QAP", "NN", "Matrix_completion"])
    parser.add_argument("--model_name", type=str, default="MCLR",  choices=["CNN", "MCLR", "DNN"])
    parser.add_argument("--times", type=int, default=1 )
    parser.add_argument("--fl_algorithm", type=str, default= "FedAvg")
    parser.add_argument("--optimizer", type=str, default="GD", choices=["FW","GD", "SGD", "PGD", "PSGD"])
    parser.add_argument("--eta_type", type=str, default="constant_eta", choices=["constant_eta", "time_varing_eta"] )
    parser.add_argument("--lambda_type", type=str, default="constant_lambda", choices=["constant_lambda", "time_varing_lambda"])

    parser.add_argument("--eta_0", type=float, default=0.05, help="step size")
    parser.add_argument("--lambda_0", type=float, default=0.01)
    parser.add_argument("--kappa", type=float,  default=10.0)
    parser.add_argument("--glob_iters", type=int, default=10)
    parser.add_argument("--local_iters", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=124)
    parser.add_argument("--device", type=int, default=0, choices=[0,1,2,3,4,5,6,7,8] )
    parser.add_argument("--fl_aggregator", type=str, default="simple_averaging", choices = ["simple_averaging", "weighted_averaging"])
    parser.add_argument("--num_users", type=int, default=10, help="should be multiple of 10") 
    parser.add_argument("--num_labels", type=int, default=10)  
    parser.add_argument("--iid", type=int, default=1, choices=[0, 1], help="0 : for iid , 1 : non-iid")

    args = parser.parse_args()


    print("=" * 60)
    print("Summary of training process:")
    print("FL Algorithm: {}".format(args.fl_algorithm))
    print("model: {}".format(args.model_name))
    print("optimizer: {}".format(args.optimizer))
    print("Aggregator: {}".format(args.fl_aggregator))
    print("eta_0: {}".format(args.eta_0))
    print("eta_type: {}".format(args.eta_type))
    print("lambda_0: {}".format(args.lambda_0))
    print("lambda_type: {}".format(args.lambda_type))
    print("kappa: {}".format(args.kappa))
    print("Batch size: {}".format(args.batch_size))
    print("Global_iters: {}".format(args.glob_iters))
    print("Local_iters: {}".format(args.local_iters))
    print("experiments: {}".format(args.times))
    print("device : {}".format(args.device))
    print("=" * 60)



    main(dataset=args.dataset,
        problem_type=args.problem_type,
        model_name=args.model_name,
        fl_algorithm=args.fl_algorithm,
        optimizer=args.optimizer,
        fl_aggregator = args.fl_aggregator,
        eta_0=args.eta_0,
        lambda_0 = args.lambda_0,
        eta_type=args.eta_type,
        lambda_type=args.lambda_type, 
        kappa=args.kappa,
        global_iters=args.glob_iters,
        local_iters=args.local_iters,
        batch_size = args.batch_size,
        times=args.times,
        gpu = args.device,
        num_users=args.num_users,
        num_labels=args.num_labels)

    
        




