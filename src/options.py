import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "FMNIST", "CIFAR10", "EMNIST", "CIFAR100", "CELEBA", "SYNTHETIC", "MOVIELENS_1m", "MOVIELENS_100k"])
    parser.add_argument("--problem_type", type=str, default="NN", choices=["QAP", "NN", "Matrix_completion"])
    parser.add_argument("--model_name", type=str, default="CNN",  choices=["CNN", "MCLR", "DNN"])
    parser.add_argument("--exp_no", type=int, default=0)
    parser.add_argument("--times", type=int, default=1)
    parser.add_argument("--fl_algorithm", type=str, default= "FedFW_Plus", choices=["FedFW","FedFW_Plus","FedAvg","FedDR", "FedProx"])
    parser.add_argument("--optimizer", type=str, default="FW", choices=["FW","GD", "SGD", "PGD", "PSGD", "PerturbedSGD"])
    parser.add_argument("--eta_type", type=str, default="time_varing_eta", choices=["constant_eta", "time_varing_eta"] )
    parser.add_argument("--lambda_type", type=str, default="time_varing_lambda", choices=["constant_lambda", "time_varing_lambda"])
    parser.add_argument("--alpha", type=float, default=0.05, help="learning rate for FedDR")
    parser.add_argument("--eta_0", type=float, default=0.001, help="step size")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate for FedAvg")
    
    parser.add_argument("--lambda_0", type=float, default=0.001)
    parser.add_argument("--kappa", type=float,  default=10.0)
    parser.add_argument("--global_iters", type=int, default=1000)
    parser.add_argument("--local_iters", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--all_batch", type=int, default=1)
    parser.add_argument("--lmo", type=str, default="lmo_l1", choices=["lmo_l1","lmo_l2"])
    parser.add_argument("--gpu", type=int, default=0, choices=[0,1,2,3,4,5,6,7,8] )
    parser.add_argument("--fl_aggregator", type=str, default="simple_averaging", choices = ["simple_averaging", "weighted_averaging"])
    parser.add_argument("--num_users", type=int, default=10, help="should be multiple of 10") 
    parser.add_argument("--num_users_perGR", type=int, default=10, help="should be <= num_users")
    parser.add_argument("--num_labels", type=int, default=10)  
    parser.add_argument("--iid", type=int, default=1, choices=[0, 1], help="0 : for iid , 1 : non-iid")

    args = parser.parse_args()

    return args
