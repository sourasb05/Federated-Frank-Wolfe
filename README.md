# The repository contains code base for different federated learning algorithms and their implementations in python. 

Algorithms
1) FedAvg 
2) FedDR
3) FedFW
4) FedFW+

# Implemented Models
non-convex settings
1) CNN
2) DNN

Convex settings
1) MCLR

# Datasets
1) Mnist
2) FMnist
3) Cifar10
4) Cifar100 
5) FEMNIST 
4) Synthetic datasets 


# Optimizers

1) Gradient descent
2) Stochastic Gradient Descent
3) Projected Gradient Descent

# Requirement

```
conda create --name personalized_fl python==3.11
conda install -c anaconda numpy scipy pandas h5py
conda install matplotlib
conda install -c pytorch torchvision
conda install -c conda-forge tqdm
```



# How to run

If you are using berzelius gpus then first you have to start interactive session.

```
interactive --gpus= number_of_gpus -t hh:mm:sc --account=account_name
```

example :

```
interactive --gpus=1 -t 01:00:00 --account=berzelius-2023-106
```

then activate the conda environment

```
activate personalized_fl
```

Now type the following to train the MCLR model with MNIST dataset and FedFW:

```
python main.py --dataset=MNIST --model=MCLR 
```

## Deep leakage privacy sub-experiment

To run the deep leakage from gradients experiment after training the model, use the option `--run_dlg`. To run the deep leakage from step directions experiment after training the model, use the option `--run_dls`. To specify the batch size for this experiment, use the `--dlg_batch_size` option. The experiment will try to create a random batch of images with size equal to the specified batch size and try to reconstruct them using gradients and step directions, respectively.


For example to run the deep leakage from gradients and deep leakage from step directions experiment with a batch size of 2, use the following:

```
python main.py --dataset=MNIST --model=MCLR --run_dlg --run_dls --dls_batch_size=2
```


