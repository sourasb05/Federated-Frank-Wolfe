python main.py --p=1 --fl_algorithm=FedDR --optimizer=PerturbedSGD --dataset=MNIST --model_name=MCLR --num_labels=3 --kappa=10
python main.py --p=1 --fl_algorithm=FedDR --optimizer=PerturbedSGD --dataset=MNIST --model_name=MCLR --num_labels=10 --kappa=10
python main.py --p=1 --fl_algorithm=FedAvg --optimizer=PGD --dataset=MNIST --model_name=MCLR --num_labels=3 --kappa=10
python main.py --p=1 --fl_algorithm=FedAvg --optimizer=PGD --dataset=MNIST --model_name=MCLR --num_labels=10 --kappa=10

python main.py --p=1 --fl_algorithm=FedDR --optimizer=PerturbedSGD --dataset=MNIST --model_name=CNN --num_labels=3 --kappa=10
python main.py --p=1 --fl_algorithm=FedDR --optimizer=PerturbedSGD --dataset=MNIST --model_name=CNN --num_labels=10 --kappa=10
python main.py --p=1 --fl_algorithm=FedAvg --optimizer=PGD --dataset=MNIST --model_name=CNN --num_labels=3 --kappa=10
python main.py --p=1 --fl_algorithm=FedAvg --optimizer=PGD --dataset=MNIST --model_name=CNN --num_labels=10 --kappa=10

python main.py --p=1 --fl_algorithm=FedDR --optimizer=PerturbedSGD --dataset=SYNTHETIC --model_name=MCLR --num_labels=3 --kappa=10
python main.py --p=1 --fl_algorithm=FedDR --optimizer=PerturbedSGD --dataset=SYNTHETIC --model_name=MCLR --num_labels=10 --kappa=10
python main.py --p=1 --fl_algorithm=FedAvg --optimizer=PGD --dataset=SYNTHETIC --model_name=MCLR --num_labels=3 --kappa=10
python main.py --p=1 --fl_algorithm=FedAvg --optimizer=PGD --dataset=SYNTHETIC --model_name=MCLR --num_labels=10 --kappa=10
 
python main.py --p=1 --fl_algorithm=FedDR --optimizer=PerturbedSGD --dataset=SYNTHETIC --model_name=DNN --num_labels=3 --kappa=10
python main.py --p=1 --fl_algorithm=FedDR --optimizer=PerturbedSGD --dataset=SYNTHETIC --model_name=DNN --num_labels=10 --kappa=10
python main.py --p=1 --fl_algorithm=FedAvg --optimizer=PGD --dataset=SYNTHETIC --model_name=DNN --num_labels=3 --kappa=10
python main.py --p=1 --fl_algorithm=FedAvg --optimizer=PGD --dataset=SYNTHETIC --model_name=DNN --num_labels=10 --kappa=10