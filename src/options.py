import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="SYNTHETIC", choices=["MNIST", "FMNIST", "CIFAR10", "EMNIST", "CIFAR100", "CELEBA", "SYNTHETIC", "MOVIELENS_1m", "MOVIELENS_100k"])
    parser.add_argument("--problem_type", type=str, default="NN", choices=["QAP", "NN", "Matrix_completion"])
    parser.add_argument("--model_name", type=str, default="MCLR",  choices=["CNN", "MCLR", "DNN"])
    parser.add_argument("--exp_no", type=int, default=0)
    parser.add_argument("--times", type=int, default=1)
    parser.add_argument("--fl_algorithm", type=str, default= "FedFW_sto", choices=["FedFW","FedFW_Plus","FedAvg","FedDR", "FedProx", "FedFW_sto"])
    parser.add_argument("--optimizer", type=str, default="FW", choices=["FW","GD", "SGD", "PGD", "PSGD", "PerturbedSGD"])
    parser.add_argument("--eta_type", type=str, default="time_varing_eta", choices=["constant_eta", "time_varing_eta"] )
    parser.add_argument("--lambda_type", type=str, default="time_varing_lambda", choices=["constant_lambda", "time_varing_lambda"])

    parser.add_argument("--split_method", type=str, default="digits", choices=["digits", "byclass"]) # for EMNIST dataset
    parser.add_argument("--total_labels", type=int, default=10, choices=[62, 10])   # for EMNIST dataset

    parser.add_argument("--alpha", type=float, default=0.01, help="learning rate for FedDR")
    parser.add_argument("--eta_0", type=float, default=0.01, help="step size")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate for FedAvg")
    
    parser.add_argument("--lambda_0", type=float, default=0.001)
    parser.add_argument("--kappa", type=float,  default=10.0)
    parser.add_argument("--global_iters", type=int, default=300)
    parser.add_argument("--local_iters", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--all_batch", type=int, default=1)
    parser.add_argument("--lmo", type=str, default="lmo_l2", choices=["lmo_l1","lmo_l2"])
    parser.add_argument("--p", type=int, default=2, help="1: l1 norm, 2: l2 norm")
    parser.add_argument("--gpu", type=int, default=0, choices=[0,1,2,3,4,5,6,7,8] )
    parser.add_argument("--fl_aggregator", type=str, default="simple_averaging", choices = ["simple_averaging", "weighted_averaging"])
    parser.add_argument("--num_users", type=int, default=100, help="should be multiple of 10") 
    parser.add_argument("--num_users_perGR", type=int, default=100, help="should be <= num_users")
    parser.add_argument("--num_labels", type=int, default=3)  
    parser.add_argument("--iid", type=int, default=1, choices=[0, 1], help="0 : for iid , 1 : non-iid")
    parser.add_argument("--run_dlg", action="store_true", help="Run deep leakage from gradient experiment")
    parser.add_argument("--run_dls", action="store_true", help="Run deep leakage from FW step dir experiment")
    parser.add_argument("--dlg_batch_size", type=int, default=8, help="Batch size for the deep leakage experiments.")

    args = parser.parse_args()

    return args
