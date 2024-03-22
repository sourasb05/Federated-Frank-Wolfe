python main.py --fl_algorithm=FedFW_sto --lmo=lmo_l2 --lambda_0=0.0001 --optimizer=FW  --split_method=digits --total_labels=10 --num_labels=10
python main.py --fl_algorithm=FedFW_sto --lmo=lmo_l1 --lambda_0=0.01 --optimizer=FW  --split_method=digits --total_labels=10 --num_labels=10
python main.py --fl_algorithm=FedFW_sto --lmo=lmo_l2 --lambda_0=0.0001 --optimizer=FW  --split_method=byclass --total_labels=62 --num_labels=62
python main.py --fl_algorithm=FedFW_sto --lmo=lmo_l1 --lambda_0=0.0001 --optimizer=FW  --split_method=byclass --total_labels=62 --num_labels=62

python main.py --fl_algorithm=FedFW_sto --lmo=lmo_l2 --lambda_0=0.0001 --optimizer=FW  --split_method=digits --total_labels=10 --num_labels=3
python main.py --fl_algorithm=FedFW_sto --lmo=lmo_l1 --lambda_0=0.0001 --optimizer=FW  --split_method=digits --total_labels=10 --num_labels=3
python main.py --fl_algorithm=FedFW_sto --lmo=lmo_l2 --lambda_0=0.0001 --optimizer=FW  --split_method=byclass --total_labels=62 --num_labels=20
python main.py --fl_algorithm=FedFW_sto --lmo=lmo_l1 --lambda_0=0.0001 --optimizer=FW  --split_method=byclass --total_labels=62 --num_labels=20


python main.py --fl_algorithm=FedFW_sto --lmo=lmo_l2 --lambda_0=0.001 --optimizer=FW --num_labels=10
python main.py --fl_algorithm=FedFW_sto --lmo=lmo_l1 --lambda_0=0.001 --optimizer=FW  --num_labels=10
python main.py --fl_algorithm=FedFW_sto --lmo=lmo_l2 --lambda_0=0.001 --optimizer=FW  --num_labels=3
python main.py --fl_algorithm=FedFW_sto --lmo=lmo_l1 --lambda_0=0.001 --optimizer=FW  --num_labels=3

python main.py --fl_algorithm=FedFW_sto --lmo=lmo_l2 --lambda_0=0.001 --optimizer=FW   --num_labels=10 --model_name=DNN --eta_type="constant_eta" --lambda_type="constant_lambda"
python main.py --fl_algorithm=FedFW_sto --lmo=lmo_l1 --lambda_0=0.001 --optimizer=FW   --num_labels=10 --model_name=DNN --eta_type="constant_eta" --lambda_type="constant_lambda"
python main.py --fl_algorithm=FedFW_sto --lmo=lmo_l2 --lambda_0=0.001 --optimizer=FW   --num_labels=3 --model_name=DNN --eta_type="constant_eta" --lambda_type="constant_lambda"
python main.py --fl_algorithm=FedFW_sto --lmo=lmo_l1 --lambda_0=0.001 --optimizer=FW   --num_labels=3 --model_name=DNN --eta_type="constant_eta" --lambda_type="constant_lambda"









python main.py --fl_algorithm=FedDR --p=2 --optimizer=PerturbedSGD  --num_labels=10 
python main.py --fl_algorithm=FedDR --p=1 --optimizer=PerturbedSGD  --num_labels=10
python main.py --fl_algorithm=FedDR --p=2 --optimizer=PerturbedSGD  --num_labels=3
python main.py --fl_algorithm=FedDR --p=1 --optimizer=PerturbedSGD  --num_labels=3

python main.py --fl_algorithm=FedDR --p=2 --optimizer=PerturbedSGD --split_method=digits --total_labels=10 --num_labels=3 
python main.py --fl_algorithm=FedDR --p=1 --optimizer=PerturbedSGD --split_method=digits --total_labels=10 --num_labels=3
python main.py --fl_algorithm=FedDR --p=2 --optimizer=PerturbedSGD --split_method=byclass --total_labels=62 --num_labels=20
python main.py --fl_algorithm=FedDR --p=1 --optimizer=PerturbedSGD --split_method=byclass --total_labels=62 --num_labels=20



python main.py --dataset=EMNIST --model_name=MCLR --fl_algorithm=FedFW --lmo=lmo_l1 --lambda_0=0.001 --global_iters=10000 --num_users=100 --optimizer=FW

python main.py --dataset=EMNIST --model_name=CNN --fl_algorithm=FedFW --lmo=lmo_l1 --eta_0=0.01 --lambda_0=0.001 --global_iters=5000 --num_users=30 --optimizer=FW
python main.py --dataset=EMNIST --model_name=CNN --fl_algorithm=FedFW --lmo=lmo_l2 --eta_0=0.01 --lambda_0=0.001 --global_iters=5000 --num_users=30 --optimizer=FW

python main.py --dataset=EMNIST --model_name=CNN --fl_algorithm=FedFW_Plus --lmo=lmo_l1 --eta_0=0.01 --lambda_0=0.001 --global_iters=5000 --num_users=30 --optimizer=FW

python main.py --dataset=EMNIST --model_name=CNN --fl_algorithm=FedDR --p=2 --eta_0=0.01 --lambda_0=0.01 --global_iters=5000 --num_users=30 --optimizer=PerturbedSGD

python main.py --dataset=EMNIST --model_name=CNN --fl_algorithm=FedDR --p=2 --eta_0=0.01 --lambda_0=0.01 --global_iters=5000 --num_users=30 --optimizer=PerturbedSGD
