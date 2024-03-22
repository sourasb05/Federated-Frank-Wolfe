import os
import h5py
import numpy as np
attributes = []
path = '/proj/sourasb-220503/codebase/FedFWplus/results/FedFW_Plus/MNIST/CNN/perf/lmo_l2/10/'
hf = h5py.File(path + '0_dataset_MNIST_fl_algorithm_FedDR_model_CNN_eta_0.001_kappa_10.0_global_iters_1000_12_03_2024'+'.h5','r')
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
fw_gap = hf.get('fw_gap')
lamda = hf.get('lambda_0')
print("eta 0", np.array(eta))
print("lamda 0", np.array(lamda))
print(f"frank wolfe gap: {np.array(fw_gap)}")
#print("train accuracy",np.array(tra))
print("test accuracy",np.max(np.array(tsa)))
#print("train loss",np.array(trl))
#print("test loss",np.array(tsl))
# print(np.array(gtra))


# print("maximum test accuracy test global :",max_acc_test_global)
# n1 = hf.get('server_aggregation_test_accuracy')
# n1 = np.array(n1)
#print(len(n1))
#print(n1[:100])