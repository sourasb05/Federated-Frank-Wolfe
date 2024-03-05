
from src.train_model.define_model_and_loss import define_model_and_loss
from src.server.FedAvg_server import Fed_Avg_Server
from src.server.FedFW_server import FedFW_Server
from src.client.FedAvg_client import Fed_Avg_Client
from src.server.FedDR_server import FedDR_Server
from src.server.FedProx_server import FedProx_Server
from src.utils.utils import select_users, read_data, read_user_data #, RMSE_nuclear
from src.MatrixComp import My_matrix_comp
from src.options import args_parser
import torch
import torch.nn as nn 
import os

def main(args):
    

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else "cpu")
    current_directory = os.getcwd()
    print(current_directory)
    
    if args.problem_type == "MATRIX_COMP": problem_category = 1
    elif args.problem_type == "QAP": problem_category = 2
    else: problem_category = 3    
    
    while args.exp_no < args.times:

        if args.problem_type == "NN":

            model, loss = define_model_and_loss(args, device)
            print(model)
            print(loss)
            
            print(args.dataset)
              
            if args.fl_algorithm == "FedFW":
                server = FedFW_Server(args, model, loss, device)
                
            elif args.fl_algorithm == "FedAvg":
                server = Fed_Avg_Server(args, model, loss, device)

            elif args.fl_algorithm == "FedDR":
                server = FedDR_Server(args, model, loss, device)
            elif args.fl_algorithm == "FedProx":
                server = FedProx_Server(args, model, loss, device)
            # server.load_model()
            server.train()
            # server.save_file()
            #server.plot_result()

        args.exp_no+=1 
    
    
    
if __name__ == "__main__":
    args = args_parser()

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
    print("Global_iters: {}".format(args.global_iters))
    print("Local_iters: {}".format(args.local_iters))
    print("experiments: {}".format(args.times))
    print("device : {}".format(args.gpu))
    print("run_dlg: {}".format(args.run_dlg))
    print("dlg_batch_size: {}".format(args.dlg_batch_size))
    print("=" * 60)

    main(args)




