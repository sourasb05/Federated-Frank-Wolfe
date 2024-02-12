import torch.nn as nn 
from src.train_model.model import *
from src.utils.utils import read_data
from src.MatrixComp import My_matrix_comp


def define_model_and_loss(args, device):

    if args.problem_type == "MATRIX_COMP":

        
        
        if  args.dataset == "MOVIELENS_100k" or args.dataset == "MOVIELENS_1m":
                data = read_data(args.dataset)
                # print(data)
                r = My_matrix_comp.matrix_comp(data)
                # input("press")
            
                print("code building in progress....")
    
    else:
            
        if args.model_name == "CNN":
                
            if args.dataset == "MNIST":
                model = cnn_Mnist().to(device)
                loss = nn.CrossEntropyLoss()

                # loss = nn.NLLLoss()
                        
            elif args.dataset == "FMNIST":
                model = cnn_Fmnist(10).to(device)
                loss = nn.CrossEntropyLoss()

            elif args.dataset == "CIFAR10":
                model = cnn_Cifar10().to(device)
                loss = nn.NLLLoss()
                    
                    
            elif args.dataset == "EMNIST":
                model = cnn_Emnist().to(device)
                loss = nn.NLLLoss()
                    
            elif args.dataset == "CELEBA":
                model = cnn_Celeba().to(device)
                loss = nn.NLLLoss()

            elif args.dataset == "CIFAR100":
                model = cnn_Cifar100().to(device)
                loss = nn.CrossEntropyLoss()

                        
        elif args.model_name == "MCLR":

            if(args.dataset == "human_activity"):
                model = Mclr_Logistic(561,6).to(device)
                loss = nn.NLLLoss()
            elif(args.dataset == "gleam"):
                model = Mclr_Logistic(561,6).to(device)
                loss = nn.NLLLoss()
            elif(args.dataset == "vehicle_sensor"):
                model = Mclr_Logistic(100,2).to(device)
                loss = nn.NLLLoss()
            elif(args.dataset == "SYNTHETIC"):
                model = Mclr_Logistic(60,10).to(device)
                loss = nn.NLLLoss()
            elif(args.dataset == "EMNIST"):
                model = Mclr_Logistic(784,62).to(device)
                loss = nn.NLLLoss()
            elif(args.dataset == "FMNIST"):
                model = Mclr_Logistic(784,10).to(device)
                loss = nn.NLLLoss() 

            elif(args.dataset == "CIFAR10"):
                model = Mclr_Logistic(3072,10).to(device)
                loss = nn.NLLLoss() 
                
            elif(args.dataset == "CIFAR100"):
                model = Mclr_Logistic(3072,100).to(device)
                loss = nn.NLLLoss() 

            else:#(dataset == "Mnist"):
                model = Mclr_Logistic().to(device)
                loss = nn.CrossEntropyLoss()
                # loss = nn.NLLLoss()
                print(model)
            
        elif(args.model_name == "DNN"):
            if(args.dataset == "human_activity"):
                model = DNN(561,100,12).to(device)
                loss = nn.NLLLoss()
            elif(args.dataset == "gleam"):
                model = DNN(561,20,6).to(device)
                loss = nn.NLLLoss()
            elif(args.dataset == "vehicle_sensor"):
                model = DNN(100,20,2).to(device)
                loss = nn.NLLLoss()
            elif(args.dataset == "SYNTHETIC"):
                model = DNN(60,20,10).to(device)
                loss = nn.NLLLoss()
            elif(args.dataset == "EMNIST"):
                model = DNN(784,200,62).to(device)
                loss = nn.NLLLoss()
            else:#(dataset == "Mnist"):
                model = DNN2().to(device)
                loss = nn.NLLLoss()
        

            
            
        else:
            print(" No problem selected")

    return model, loss       
        