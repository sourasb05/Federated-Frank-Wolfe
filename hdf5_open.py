import os
import h5py
import numpy as np
attributes = []
hf = h5py.File('results/FedFW/MNIST/CNN/constant_eta/constant_lambda/perf/3/0_dataset_MNIST_aggregator_simple_averaging_fl_algorithm_FedFW_model_CNN_lamdba_0_0.09999999999999998_eta_0_0.00010000000000000003_kappa_10.0_global_iters_1000_08_02_2024.h5','r')
id = 0

for key in hf.keys():
    attributes.append(key)
    print("id [",id,"] :", key)
    id+=1
print(attributes)
tsl = hf.get('global_test_loss')
tsa = hf.get('global_test_accuracy')
trl = hf.get('global_train_loss')
tra = hf.get('global_train_accuracy')
eta = hf.get('eta_0')
lamda = hf.get('lambda_0')
print("eta 0", np.array(eta))
print("lamda 0", np.array(lamda))
#print("train accuracy",np.array(tra))
print("test accuracy",np.array(tsa))
#print("train loss",np.array(trl))
#print("test loss",np.array(tsl))
# print(np.array(gtra))


# print("maximum test accuracy test global :",max_acc_test_global)
# n1 = hf.get('server_aggregation_test_accuracy')
# n1 = np.array(n1)
#print(len(n1))
#print(n1[:100])