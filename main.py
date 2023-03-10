from model import *
from server import *
import argparse

def main(exp_no, times, model, dataset, device, algorithm ):
    exp_no=0
    while exp_no < times:
        if model == "CNN":
            if dataset == "MNIST":
                model = Net().to(device), model
            else:
                model = cnn_fmnist().to(device), model
        else:
            print(" do nothing")
        
        if algorithm == "FedAvg":
            server = FedAvg()

        server.train()

        exp_no+=1 
    
    if __name__ == "__main__":
        parser = argparse.ArgumentParser()

        parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "FMnist"])
        parser.add_argument("--model", type=str, default="CNN")
        parser.add_argument("--times", type=int, default=1 )
        parser.add_argument("--algorithm", type=str, default= "FedAvg")
        parser.add_argument("--optimizer", type=str, default="GD", choices=["GD", "SGD", "FW"])
        parser.add_argument("--step_size" type=double, default=0.05)
        parser.add_argument("--glob_iters", type=int, defaul=10)
        parser.add_argument("--local_iters", type=int, default=1)
        parser.add_argument("--batch_size", type=int, default=124)
        

        args = parser.parse_args()


    print("=" * 90)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("model: {}".format(args.model))
    print("optimizer: {}".format(args.optimizer))
    print("Step_size: {}".format(args.step_size))
    print("Global_iters: {}".format(args.glob_iters))
    print("Local_iters: {}".format(args.local_iters))
    print("experiments: {}".format(args.times))
    print("=" * 90)

    main(dataset=args.dataset,
        model=args.model,
        times=args.times,
        algorithm=args.algorithm,
        optimizer=args.optimizer,
        step_size=args.step_size,
        glob_iters=args.glob_iters,
        local_iters=args.local_iters
        )

    
        




