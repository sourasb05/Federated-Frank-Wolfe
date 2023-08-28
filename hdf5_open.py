import os
import h5py
import numpy as np
attributes = []
hf = h5py.File('/proj/sourasb-220503/codebase/FedFWplus/results/FedFW/MNIST/MCLR/0_dataset_MNIST_aggregator_simple_averaging_fl_algorithm_FedFW_model_MCLR_lamdba_01e-05_kappa_1.0_global_iters_1028_08_2023.h5','r')
id = 0

for key in hf.keys():
    attributes.append(key)
    print("id [",id,"] :", key)
    id+=1
print(attributes)
tsl = hf.get('global_test_loss')
trl = hf.get('global_test_accuracy')
tra = hf.get('global_train_loss')
tsa = hf.get('global_train_accuracy')

print("train accuracy",np.array(tra))
print("test accuracy",np.array(tsa))
print("train loss",np.array(trl))
print("test loss",np.array(tsl))
# print(np.array(gtra))


# print("maximum test accuracy test global :",max_acc_test_global)
# n1 = hf.get('server_aggregation_test_accuracy')
# n1 = np.array(n1)
#print(len(n1))
#print(n1[:100])