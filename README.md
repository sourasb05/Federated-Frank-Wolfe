# The repository contains the codes for the paper Federated Frank-Wolfe


Algorithms
1) FedFW
2) FedFW+
3) FedFW-stochastics 
2) FedDR

# Implemented Models
non-convex settings
1) CNN
2) DNN

Convex settings
1) MCLR

# Datasets
1) Mnist
3) Cifar10
5) EMNIST-10
6) EMNIST-62 
4) Synthetic datasets 


# Optimizers
1) FedFW
2) FedFW+
3) FedFW-stochastics
4) PerturbedSGD

# Requirement

```
conda create --name personalized_fl python==3.11
conda install -c anaconda numpy scipy pandas h5py
conda install matplotlib
conda install -c pytorch torchvision
conda install -c conda-forge tqdm
```
# How to run

To train the a MCLR model using FedFW algorithm on dataset MNIST non_iid :

```
python main.py --dataset=MNIST --model=MCLR  --lambda_0=0.001 --lmo=lmo_l2 --num_labels=3 --num_users=100 --num_users_perGR=100
```

## Deep leakage privacy sub-experiment

To run the deep leakage from gradients experiment after training the model, use the option `--run_dlg`. To run the deep leakage from step directions experiment after training the model, use the option `--run_dls`. To specify the batch size for this experiment, use the `--dlg_batch_size` option. The experiment will try to create a random batch of images with size equal to the specified batch size and try to reconstruct them using gradients and step directions, respectively.


For example to run the deep leakage from gradients and deep leakage from step directions experiment with a batch size of 2, use the following:

```
python main.py --dataset=MNIST --model=MCLR --run_dlg --run_dls --dls_batch_size=2
```


## Acknowledgement

The base framework is inspired from https://github.com/CharlieDinh/pFedMe/tree/master
