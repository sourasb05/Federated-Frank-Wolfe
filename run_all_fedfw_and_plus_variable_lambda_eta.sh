python main.py --fl_algorithm=FedFW --dataset=MNIST --model_name=MCLR --eta_0=0.1 --lambda_0=0.1 --num_labels=3 --lmo=lmo_l1 --kappa=10
python main.py --fl_algorithm=FedFW --dataset=MNIST --model_name=MCLR --eta_0=0.1 --lambda_0=0.1 --num_labels=10 --lmo=lmo_l1 --kappa=10
python main.py --fl_algorithm=FedFW_Plus --dataset=MNIST --model_name=MCLR --eta_0=0.1 --lambda_0=0.1 --num_labels=3 --lmo=lmo_l1 --kappa=10
python main.py --fl_algorithm=FedFW_Plus --dataset=MNIST --model_name=MCLR --eta_0=0.1 --lambda_0=0.1 --num_labels=10 --lmo=lmo_l1 --kappa=10

python main.py --fl_algorithm=FedFW --dataset=MNIST --model_name=CNN --eta_0=0.1 --lambda_0=0.1 --num_labels=3 --lmo=lmo_l1 --kappa=10
python main.py --fl_algorithm=FedFW --dataset=MNIST --model_name=CNN --eta_0=0.1 --lambda_0=0.1 --num_labels=10 --lmo=lmo_l1 --kappa=10
python main.py --fl_algorithm=FedFW_Plus --dataset=MNIST --model_name=CNN --eta_0=0.1 --lambda_0=0.1 --num_labels=3 --lmo=lmo_l1 --kappa=10
python main.py --fl_algorithm=FedFW_Plus --dataset=MNIST --model_name=CNN --eta_0=0.1 --lambda_0=0.1 --num_labels=10 --lmo=lmo_l1 --kappa=10
python main.py --fl_algorithm=FedFW --dataset=SYNTHETIC --model_name=MCLR --eta_0=0.1 --lambda_0=0.1 --num_labels=3 --lmo=lmo_l1 --kappa=10
python main.py --fl_algorithm=FedFW --dataset=SYNTHETIC --model_name=MCLR --eta_0=0.1 --lambda_0=0.1 --num_labels=10 --lmo=lmo_l1 --kappa=10
python main.py --fl_algorithm=FedFW_Plus --dataset=SYNTHETIC --model_name=MCLR --eta_0=0.1 --lambda_0=0.1 --num_labels=3 --lmo=lmo_l1 --kappa=10
python main.py --fl_algorithm=FedFW_Plus --dataset=SYNTHETIC --model_name=MCLR --eta_0=0.1 --lambda_0=0.1 --num_labels=10 --lmo=lmo_l1 --kappa=10

python main.py --fl_algorithm=FedFW --dataset=SYNTHETIC --model_name=DNN --eta_0=0.1 --lambda_0=0.1 --num_labels=3 --lmo=lmo_l1 --kappa=10
python main.py --fl_algorithm=FedFW --dataset=SYNTHETIC --model_name=DNN --eta_0=0.1 --lambda_0=0.1 --num_labels=10 --lmo=lmo_l1 --kappa=10
python main.py --fl_algorithm=FedFW_Plus --dataset=SYNTHETIC --model_name=DNN --eta_0=0.1 --lambda_0=0.1 --num_labels=3 --lmo=lmo_l1 --kappa=10
python main.py --fl_algorithm=FedFW_Plus --dataset=SYNTHETIC --model_name=DNN --eta_0=0.1 --lambda_0=0.1 --num_labels=10 --lmo=lmo_l1 --kappa=10
