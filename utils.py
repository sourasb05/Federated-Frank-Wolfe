import json
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import trange
import random

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CHANNELS = 1


IMAGE_SIZE_CIFAR = 32
NUM_CHANNELS_CIFAR = 3

""""
def suffle_data(data):
    data_x = data['x']
    data_y = data['y']
        # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)
    return (data_x, data_y)




def get_batch_sample(data, batch_size):
    ""data_x = data['x']
    data_y = data['y']

    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    batched_x = data_x[0:batch_size]
    batched_y = data_y[0:batch_size]
    return (batched_x, batched_y)

def get_random_batch_sample(data_x, data_y, batch_size):
    num_parts = len(data_x)//batch_size + 1
    if(len(data_x) > batch_size):
        batch_idx = np.random.choice(list(range(num_parts +1)))
        sample_index = batch_idx*batch_size
        if(sample_index + batch_size > len(data_x)):
            return (data_x[sample_index:], data_y[sample_index:])
        else:
            return (data_x[sample_index: sample_index+batch_size], data_y[sample_index: sample_index+batch_size])
    else:
        return (data_x,data_y)


Check if code goes through this function

"""

def read_cifar10_data():
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data),shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data),shuffle=False)

    """
    enumerate(iterable, start=0)
    iterable : Any object that supports iterations
    start : The index value from where  the iteration will be started. Default is 0

    """
    for _, train_data in enumerate(trainloader,0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader,0):
        testset.data, testset.targets = test_data

    random.seed(1)
    np.random.seed(1)
    NUM_USERS = 10 # should be muitiple of 10
    NUM_LABELS = 3
    # Setup directory for train/test data
    train_path = './data/train/cifa_train_100.json'
    test_path = './data/test/cifa_test_100.json'
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    cifa_data_image = []
    cifa_data_label = []
    """
    detach() : Pytorch is a Python and C++ interface for an open-source deep learning platform.
    It is found within the torch module.
    In PyTorch, the input data has to be processed in the form of a tensor. 
    It also includes a module that calculates gradients automatically for backpropagation. 
    Tensor.detach() method in PyTorch is used to separate a tensor from the computational graph 
    by returning a new tensor that doesn’t require a gradient. If we want to move a tensor from
    the Graphical Processing Unit (GPU) to the Central Processing Unit (CPU), then we can use 
    detach() method. It will not take any parameter and return the detached tensor.

    extend() : The extend() method adds all the elements of an iterable (list, tuple, string etc.)
      to the end of the list
    """
    cifa_data_image.extend(trainset.data.cpu().detach().numpy()) 
    cifa_data_image.extend(testset.data.cpu().detach().numpy())
    cifa_data_label.extend(trainset.targets.cpu().detach().numpy())
    cifa_data_label.extend(testset.targets.cpu().detach().numpy())
    cifa_data_image = np.array(cifa_data_image)
    cifa_data_label = np.array(cifa_data_label)

    cifa_data = []
    for i in trange(10):
        idx = cifa_data_label==i
        cifa_data.append(cifa_data_image[idx])


    # print("\nNumb samples of each label:\n", [len(v) for v in cifa_data])
    users_lables = []


    ###### CREATE USER DATA SPLIT #######
    # Assign 100 samples to each user
    X = [[] for _ in range(NUM_USERS)]
    y = [[] for _ in range(NUM_USERS)]
    idx = np.zeros(10, dtype=np.int64)
    for user in range(NUM_USERS):
        for j in range(NUM_LABELS):  # 3 labels for each users
            #l = (2*user+j)%10
            l = (user + j) % 10
            # print("L:", l)
            X[user] += cifa_data[l][idx[l]:idx[l]+10].tolist()
            y[user] += (l*np.ones(10)).tolist()
            idx[l] += 10

    # print("IDX1:", idx)  # counting samples for each labels

    # Assign remaining sample by power law
    user = 0
    props = np.random.lognormal(
        0, 2., (10, NUM_USERS, NUM_LABELS))  
    props = np.array([[[len(v)-NUM_USERS]] for v in cifa_data]) * \
        props/np.sum(props, (1, 2), keepdims=True)
    # print("here:",props/np.sum(props,(1,2), keepdims=True))
    #props = np.array([[[len(v)-100]] for v in mnist_data]) * \
    #    props/np.sum(props, (1, 2), keepdims=True)
    #idx = 1000*np.ones(10, dtype=np.int64)
    # print("here2:",props)
    for user in trange(NUM_USERS):
        for j in range(NUM_LABELS):  # 4 labels for each users
            # l = (2*user+j)%10
            l = (user + j) % 10
            num_samples = int(props[l, user//int(NUM_USERS/10), j])
            numran1 = random.randint(300, 600)
            num_samples = (num_samples)  + numran1 #+ 200
            if(NUM_USERS <= 20): 
                num_samples = num_samples * 2
            if idx[l] + num_samples < len(cifa_data[l]):
                X[user] += cifa_data[l][idx[l]:idx[l]+num_samples].tolist()
                y[user] += (l*np.ones(num_samples)).tolist()
                idx[l] += num_samples
                # print("check len os user:", user, j, "len data", len(X[user]), num_samples)

    # print("IDX2:", idx) # counting samples for each labels

    # Create data structure
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

    # Setup 5 users
    # for i in trange(5, ncols=120):
    for i in range(NUM_USERS):
        uname = i
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)

        num_samples = len(X[i])
        train_len = int(0.75*num_samples)
        test_len = num_samples - train_len

        #X_train, X_test, y_train, y_test = train_test_split(X[i], y[i], train_size=0.75, stratify=y[i])\
        
        test_data['users'].append(uname)
        test_data["user_data"][uname] =  {'x': X[i][:test_len], 'y': y[i][:test_len]} 
        test_data['num_samples'].append(test_len)

        train_data["user_data"][uname] =  {'x': X[i][test_len:], 'y': y[i][test_len:]}
        train_data['users'].append(uname)
        train_data['num_samples'].append(train_len)
        
    return train_data['users'], train_data['user_data'], test_data['user_data']

def read_cifar100_data():
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data),shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data),shuffle=False)

    """
    enumerate(iterable, start=0)
    iterable : Any object that supports iterations
    start : The index value from where  the iteration will be started. Default is 0

    """
    for _, train_data in enumerate(trainloader,0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader,0):
        testset.data, testset.targets = test_data

    random.seed(1)
    np.random.seed(1)
    NUM_USERS = 1000 # should be muitiple of 10
    NUM_LABELS = 3
    # Setup directory for train/test data
    train_path = './data/train/cifar100_train.json'
    test_path = './data/test/cifar100_test.json'
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    cifar100_data_image = []
    cifar100_data_label = []
    
    cifar100_data_image.extend(trainset.data.cpu().detach().numpy()) 
    cifar100_data_image.extend(testset.data.cpu().detach().numpy())
    cifar100_data_label.extend(trainset.targets.cpu().detach().numpy())
    cifar100_data_label.extend(testset.targets.cpu().detach().numpy())
    cifar100_data_image = np.array(cifar100_data_image)
    cifar100_data_label = np.array(cifar100_data_label)

    cifar100_data = []
    for i in trange(100):
        idx = cifar100_data_label==i
        cifar100_data.append(cifar100_data_image[idx])

    #print(len(cifar100_data))
    # input("press")


    # print("\nNumb samples of each label:\n", [len(v) for v in cifa_data])
    users_lables = []


    ###### CREATE USER DATA SPLIT #######
    # Assign 100 samples to each user
    X = [[] for _ in range(NUM_USERS)]
    y = [[] for _ in range(NUM_USERS)]
    idx = np.zeros(100, dtype=np.int64)
    for user in range(NUM_USERS):
        for j in range(NUM_LABELS):  # 3 labels for each users
            #l = (2*user+j)%10
            l = (user + j) % 100
            # print("L:", l)
            X[user] += cifar100_data[l][idx[l]:idx[l]+10].tolist()
            y[user] += (l*np.ones(100)).tolist()

            # print("X[",user,"] :",X[user])
            # print("y[",user,"] :",X[user])
            # print("idx[",l,"] :",idx[l])
            # input("press")
            
            idx[l] += 10

    # print("IDX1:", idx)  # counting samples for each labels

    # Assign remaining sample by power law
    user = 0
    props = np.random.lognormal(0, 2., (100, NUM_USERS, NUM_LABELS))  
    
    props = np.array([[[len(v)-NUM_USERS]] for v in cifar100_data]) * \
        props/np.sum(props, (1, 2), keepdims=True)
    
    # print(props)


    for user in trange(NUM_USERS):
        for j in range(NUM_LABELS):  # 4 labels for each users
            # l = (2*user+j)%10
            l = (user + j) % 100
            num_samples = int(props[l, user//int(NUM_USERS/100), j])
            
            numran1 = random.randint(300, 600)
            num_samples = (num_samples)  + numran1 #+ 200
            print(" num_samples", num_samples)


            if(NUM_USERS <= 20): 
                num_samples = num_samples * 2
            print("len(cfar100_data[",l,"] :", len(cifar100_data[l]))
            if idx[l] + num_samples < len(cifar100_data[l]):
                X[user] += cifar100_data[l][idx[l]:idx[l]+num_samples].tolist()
                y[user] += (l*np.ones(num_samples)).tolist()
                idx[l] += num_samples

                print("X[",user,"] :",X[user])
                print("y[",user,"] :",X[user])
                print("idx[",l,"] :",idx[l])
                input("press")
                # print("check len os user:", user, j, "len data", len(X[user]), num_samples)

    # print("IDX2:", idx) # counting samples for each labels

    # Create data structure
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

    # Setup 5 users
    # for i in trange(5, ncols=120):
    for i in range(NUM_USERS):
        uname = i
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)

        num_samples = len(X[i])
        train_len = int(0.75*num_samples)
        test_len = num_samples - train_len

        #X_train, X_test, y_train, y_test = train_test_split(X[i], y[i], train_size=0.75, stratify=y[i])\
        
        test_data['users'].append(uname)
        test_data["user_data"][uname] =  {'x': X[i][:test_len], 'y': y[i][:test_len]} 
        test_data['num_samples'].append(test_len)

        train_data["user_data"][uname] =  {'x': X[i][test_len:], 'y': y[i][test_len:]}
        train_data['users'].append(uname)
        train_data['num_samples'].append(train_len)
        
    return train_data['users'], train_data['user_data'], test_data['user_data']

def read_FMnist_data():
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.FMNIST(root='./data', train=True,download=True, transform=transform)
    testset = torchvision.datasets.FMNIST(root='./data', train=False,download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data),shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data),shuffle=False)

    """
    enumerate(iterable, start=0)
    iterable : Any object that supports iterations
    start : The index value from where  the iteration will be started. Default is 0

    """
    for _, train_data in enumerate(trainloader,0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader,0):
        testset.data, testset.targets = test_data

    random.seed(1)
    np.random.seed(1)
    NUM_USERS = 1000 # should be muitiple of 10
    NUM_LABELS = 10
    # Setup directory for train/test data
    train_path = './data/train/mnist_train.json'
    test_path = './data/test/mnist_test.json'
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    fmnist_data_image = []
    fmnist_data_label = []
    
    fmnist_data_image.extend(trainset.data.cpu().detach().numpy()) 
    fmnist_data_image.extend(testset.data.cpu().detach().numpy())
    fmnist_data_label.extend(trainset.targets.cpu().detach().numpy())
    fmnist_data_label.extend(testset.targets.cpu().detach().numpy())
    fmnist_data_image = np.array(fmnist_data_image)
    fmnist_data_label = np.array(fmnist_data_label)

    fmnist_data = []
    for i in trange(10):
        idx = fmnist_data_label==i
        fmnist_data.append(fmnist_data_image[idx])


    # print("\nNumb samples of each label:\n", [len(v) for v in cifa_data])
    users_lables = []


    ###### CREATE USER DATA SPLIT #######
    # Assign 100 samples to each user
    X = [[] for _ in range(NUM_USERS)]
    y = [[] for _ in range(NUM_USERS)]
    idx = np.zeros(10, dtype=np.int64)
    for user in range(NUM_USERS):
        for j in range(NUM_LABELS):  # 3 labels for each users
            #l = (2*user+j)%10
            l = (user + j) % 10
            # print("L:", l)
            X[user] += fmnist_data[l][idx[l]:idx[l]+10].tolist()
            y[user] += (l*np.ones(10)).tolist()
            idx[l] += 10

    # print("IDX1:", idx)  # counting samples for each labels

    # Assign remaining sample by power law
    user = 0
    props = np.random.lognormal(
        0, 2., (10, NUM_USERS, NUM_LABELS))  # last 5 is 5 labels
    props = np.array([[[len(v)-NUM_USERS]] for v in fmnist_data]) * \
        props/np.sum(props, (1, 2), keepdims=True)
    # print("here:",props/np.sum(props,(1,2), keepdims=True))
    #props = np.array([[[len(v)-100]] for v in mnist_data]) * \
    #    props/np.sum(props, (1, 2), keepdims=True)
    #idx = 1000*np.ones(10, dtype=np.int64)
    # print("here2:",props)
    for user in trange(NUM_USERS):
        for j in range(NUM_LABELS):  # 4 labels for each users
            # l = (2*user+j)%10
            l = (user + j) % 10
            num_samples = int(props[l, user//int(NUM_USERS/10), j])
            numran1 = random.randint(300, 600)
            num_samples = (num_samples)  + numran1 #+ 200
            if(NUM_USERS <= 20): 
                num_samples = num_samples * 2
            if idx[l] + num_samples < len(fmnist_data[l]):
                X[user] += fmnist_data[l][idx[l]:idx[l]+num_samples].tolist()
                y[user] += (l*np.ones(num_samples)).tolist()
                idx[l] += num_samples
                # print("check len os user:", user, j, "len data", len(X[user]), num_samples)

    # print("IDX2:", idx) # counting samples for each labels

    # Create data structure
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

    # Setup 5 users
    # for i in trange(5, ncols=120):
    for i in range(NUM_USERS):
        uname = i
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)

        num_samples = len(X[i])
        train_len = int(0.75*num_samples)
        test_len = num_samples - train_len

        #X_train, X_test, y_train, y_test = train_test_split(X[i], y[i], train_size=0.75, stratify=y[i])\
        
        test_data['users'].append(uname)
        test_data["user_data"][uname] =  {'x': X[i][:test_len], 'y': y[i][:test_len]} 
        test_data['num_samples'].append(test_len)

        train_data["user_data"][uname] =  {'x': X[i][test_len:], 'y': y[i][test_len:]}
        train_data['users'].append(uname)
        train_data['num_samples'].append(train_len)
        
    return train_data['users'], train_data['user_data'], test_data['user_data']


def read_Mnist_data():
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5], [0.5])])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False,download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data),shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data),shuffle=False)

    """
    enumerate(iterable, start=0)
    iterable : Any object that supports iterations
    start : The index value from where  the iteration will be started. Default is 0

    """
    for _, train_data in enumerate(trainloader,0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader,0):
        testset.data, testset.targets = test_data

    random.seed(1)
    np.random.seed(1)
    NUM_USERS = 10 # should be muitiple of 10
    NUM_LABELS = 3
    # Setup directory for train/test data
    train_path = './data/train/mnist_train.json'
    test_path = './data/test/mnist_test.json'
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    mnist_data_image = []
    mnist_data_label = []
    
    mnist_data_image.extend(trainset.data.cpu().detach().numpy()) 
    mnist_data_image.extend(testset.data.cpu().detach().numpy())
    mnist_data_label.extend(trainset.targets.cpu().detach().numpy())
    mnist_data_label.extend(testset.targets.cpu().detach().numpy())
    mnist_data_image = np.array(mnist_data_image)
    mnist_data_label = np.array(mnist_data_label)

    mnist_data = []
    for i in trange(10):
        idx = mnist_data_label==i
        mnist_data.append(mnist_data_image[idx])


    # print("\nNumb samples of each label:\n", [len(v) for v in cifa_data])
    users_lables = []


    ###### CREATE USER DATA SPLIT #######
    # Assign 100 samples to each user
    X = [[] for _ in range(NUM_USERS)]
    y = [[] for _ in range(NUM_USERS)]
    idx = np.zeros(10, dtype=np.int64)
    for user in range(NUM_USERS):
        for j in range(NUM_LABELS):  # 3 labels for each users
            #l = (2*user+j)%10
            l = (user + j) % 10
            # print("L:", l)
            X[user] += mnist_data[l][idx[l]:idx[l]+10].tolist()
            y[user] += (l*np.ones(10)).tolist()
            idx[l] += 10

    # print("IDX1:", idx)  # counting samples for each labels

    # Assign remaining sample by power law
    user = 0
    props = np.random.lognormal(
        0, 2., (10, NUM_USERS, NUM_LABELS))  # last 5 is 5 labels
    props = np.array([[[len(v)-NUM_USERS]] for v in mnist_data]) * \
        props/np.sum(props, (1, 2), keepdims=True)
    # print("here:",props/np.sum(props,(1,2), keepdims=True))
    #props = np.array([[[len(v)-100]] for v in mnist_data]) * \
    #    props/np.sum(props, (1, 2), keepdims=True)
    #idx = 1000*np.ones(10, dtype=np.int64)
    # print("here2:",props)
    for user in trange(NUM_USERS):
        for j in range(NUM_LABELS):  # 4 labels for each users
            # l = (2*user+j)%10
            l = (user + j) % 10
            num_samples = int(props[l, user//int(NUM_USERS/10), j])
            numran1 = random.randint(300, 600)
            num_samples = (num_samples)  + numran1 #+ 200
            if(NUM_USERS <= 20): 
                num_samples = num_samples * 2
            if idx[l] + num_samples < len(mnist_data[l]):
                X[user] += mnist_data[l][idx[l]:idx[l]+num_samples].tolist()
                y[user] += (l*np.ones(num_samples)).tolist()
                idx[l] += num_samples
                # print("check len os user:", user, j, "len data", len(X[user]), num_samples)

    # print("IDX2:", idx) # counting samples for each labels

    # Create data structure
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

    # Setup 5 users
    # for i in trange(5, ncols=120):
    for i in range(NUM_USERS):
        uname = i
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)

        num_samples = len(X[i])
        train_len = int(0.75*num_samples)
        test_len = num_samples - train_len

        #X_train, X_test, y_train, y_test = train_test_split(X[i], y[i], train_size=0.75, stratify=y[i])\
        
        test_data['users'].append(uname)
        test_data["user_data"][uname] =  {'x': X[i][:test_len], 'y': y[i][:test_len]} 
        test_data['num_samples'].append(test_len)

        train_data["user_data"][uname] =  {'x': X[i][test_len:], 'y': y[i][test_len:]}
        train_data['users'].append(uname)
        train_data['num_samples'].append(train_len)
        
    return train_data['users'], train_data['user_data'], test_data['user_data']


def read_EMnist_data():
    transform = transforms.Compose([transforms.ToTensor(), # convert to tensor
                                    transforms.Normalize((0.1307,), (0.3081,))])  # normalize the data


    # Download the EMNIST dataset and apply the transformation

    """
    We can split 
    digits that contains 10 classes,
    byClass contains 62 classes,
    byMerge contains 47 classes,
    Balanced contains 47 classes,
    Letters contains 26 classes,

    """
    trainset = torchvision.datasets.EMNIST(root='./data', train=True,download=True, transform=transform)
    testset = torchvision.datasets.EMNIST(root='./data', train=False,download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data),shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data),shuffle=False)

    """
    enumerate(iterable, start=0)
    iterable : Any object that supports iterations
    start : The index value from where  the iteration will be started. Default is 0

    """
    for _, train_data in enumerate(trainloader,0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader,0):
        testset.data, testset.targets = test_data

    random.seed(1)
    np.random.seed(1)
    NUM_USERS = 1000 # should be muitiple of 10
    NUM_LABELS = 10
    # Setup directory for train/test data
    train_path = './data/train/emnist_train.json'
    test_path = './data/test/emnist_test.json'
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    emnist_data_image = []
    emnist_data_label = []
    
    emnist_data_image.extend(trainset.data.cpu().detach().numpy()) 
    emnist_data_image.extend(testset.data.cpu().detach().numpy())
    emnist_data_label.extend(trainset.targets.cpu().detach().numpy())
    emnist_data_label.extend(testset.targets.cpu().detach().numpy())
    emnist_data_image = np.array(emnist_data_image)
    emnist_data_label = np.array(emnist_data_label)

    emnist_data = []
    for i in trange(10):
        idx = emnist_data_label==i
        emnist_data.append(emnist_data_image[idx])


    # print("\nNumb samples of each label:\n", [len(v) for v in cifa_data])
    users_lables = []


    ###### CREATE USER DATA SPLIT #######
    # Assign 100 samples to each user
    X = [[] for _ in range(NUM_USERS)]
    y = [[] for _ in range(NUM_USERS)]
    idx = np.zeros(10, dtype=np.int64)
    for user in range(NUM_USERS):
        for j in range(NUM_LABELS):  # 3 labels for each users
            #l = (2*user+j)%10
            l = (user + j) % 10
            # print("L:", l)
            X[user] += emnist_data[l][idx[l]:idx[l]+10].tolist()
            y[user] += (l*np.ones(10)).tolist()
            idx[l] += 10

    # print("IDX1:", idx)  # counting samples for each labels

    # Assign remaining sample by power law
    user = 0
    props = np.random.lognormal(
        0, 2., (10, NUM_USERS, NUM_LABELS))  # last 5 is 5 labels
    props = np.array([[[len(v)-NUM_USERS]] for v in emnist_data]) * \
        props/np.sum(props, (1, 2), keepdims=True)
    # print("here:",props/np.sum(props,(1,2), keepdims=True))
    #props = np.array([[[len(v)-100]] for v in mnist_data]) * \
    #    props/np.sum(props, (1, 2), keepdims=True)
    #idx = 1000*np.ones(10, dtype=np.int64)
    # print("here2:",props)
    for user in trange(NUM_USERS):
        for j in range(NUM_LABELS):  # 4 labels for each users
            # l = (2*user+j)%10
            l = (user + j) % 10
            num_samples = int(props[l, user//int(NUM_USERS/10), j])
            numran1 = random.randint(300, 600)
            num_samples = (num_samples)  + numran1 #+ 200
            if(NUM_USERS <= 20): 
                num_samples = num_samples * 2
            if idx[l] + num_samples < len(emnist_data[l]):
                X[user] += emnist_data[l][idx[l]:idx[l]+num_samples].tolist()
                y[user] += (l*np.ones(num_samples)).tolist()
                idx[l] += num_samples
                # print("check len os user:", user, j, "len data", len(X[user]), num_samples)

    # print("IDX2:", idx) # counting samples for each labels

    # Create data structure
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

    # Setup 5 users
    # for i in trange(5, ncols=120):
    for i in range(NUM_USERS):
        uname = i
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)

        num_samples = len(X[i])
        train_len = int(0.75*num_samples)
        test_len = num_samples - train_len

        #X_train, X_test, y_train, y_test = train_test_split(X[i], y[i], train_size=0.75, stratify=y[i])\
        
        test_data['users'].append(uname)
        test_data["user_data"][uname] =  {'x': X[i][:test_len], 'y': y[i][:test_len]} 
        test_data['num_samples'].append(test_len)

        train_data["user_data"][uname] =  {'x': X[i][test_len:], 'y': y[i][test_len:]}
        train_data['users'].append(uname)
        train_data['num_samples'].append(train_len)
        
    return train_data['users'], train_data['user_data'], test_data['user_data']

def read_Celeba_data():
    # Define transformations to be applied to the input images
    transform = transforms.Compose([ 
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    trainset = torchvision.datasets.CelebA(root='./data', train=True,download=True, transform=transform)
    testset = torchvision.datasets.CelebA(root='./data', train=False,download=True, transform=transform)
    trainloader = torch.utils.data.CelebA(trainset, batch_size=len(trainset.data),shuffle=False)
    testloader = torch.utils.data.CelebA(testset, batch_size=len(testset.data),shuffle=False)

    """
    enumerate(iterable, start=0)
    iterable : Any object that supports iterations
    start : The index value from where  the iteration will be started. Default is 0

    """
    for _, train_data in enumerate(trainloader,0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader,0):
        testset.data, testset.targets = test_data

    random.seed(1)
    np.random.seed(1)
    NUM_USERS = 1000 # should be muitiple of 10
    NUM_LABELS = 10
    # Setup directory for train/test data
    train_path = './data/train/celeba_train.json'
    test_path = './data/test/celeba_test.json'
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    celeba_data_image = []
    celeba_data_label = []
    
    celeba_data_image.extend(trainset.data.cpu().detach().numpy()) 
    celeba_data_image.extend(testset.data.cpu().detach().numpy())
    celeba_data_label.extend(trainset.targets.cpu().detach().numpy())
    celeba_data_label.extend(testset.targets.cpu().detach().numpy())
    celeba_data_image = np.array(celeba_data_image)
    celeba_data_label = np.array(celeba_data_label)

    celeba_data = []
    for i in trange(10):
        idx = celeba_data_label==i
        celeba_data.append(celeba_data_image[idx])


    # print("\nNumb samples of each label:\n", [len(v) for v in cifa_data])
    users_lables = []


    ###### CREATE USER DATA SPLIT #######
    # Assign 100 samples to each user
    X = [[] for _ in range(NUM_USERS)]
    y = [[] for _ in range(NUM_USERS)]
    idx = np.zeros(10, dtype=np.int64)
    for user in range(NUM_USERS):
        for j in range(NUM_LABELS):  # 3 labels for each users
            #l = (2*user+j)%10
            l = (user + j) % 10
            # print("L:", l)
            X[user] += celeba_data[l][idx[l]:idx[l]+10].tolist()
            y[user] += (l*np.ones(10)).tolist()
            idx[l] += 10

    # print("IDX1:", idx)  # counting samples for each labels

    # Assign remaining sample by power law
    user = 0
    props = np.random.lognormal(
        0, 2., (10, NUM_USERS, NUM_LABELS))  # last 5 is 5 labels
    props = np.array([[[len(v)-NUM_USERS]] for v in celeba_data]) * \
        props/np.sum(props, (1, 2), keepdims=True)
    # print("here:",props/np.sum(props,(1,2), keepdims=True))
    #props = np.array([[[len(v)-100]] for v in mnist_data]) * \
    #    props/np.sum(props, (1, 2), keepdims=True)
    #idx = 1000*np.ones(10, dtype=np.int64)
    # print("here2:",props)
    for user in trange(NUM_USERS):
        for j in range(NUM_LABELS):  # 4 labels for each users
            # l = (2*user+j)%10
            l = (user + j) % 10
            num_samples = int(props[l, user//int(NUM_USERS/10), j])
            numran1 = random.randint(300, 600)
            num_samples = (num_samples)  + numran1 #+ 200
            if(NUM_USERS <= 20): 
                num_samples = num_samples * 2
            if idx[l] + num_samples < len(celeba_data[l]):
                X[user] += celeba_data[l][idx[l]:idx[l]+num_samples].tolist()
                y[user] += (l*np.ones(num_samples)).tolist()
                idx[l] += num_samples
                # print("check len os user:", user, j, "len data", len(X[user]), num_samples)

    # print("IDX2:", idx) # counting samples for each labels

    # Create data structure
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

    # Setup 5 users
    # for i in trange(5, ncols=120):
    for i in range(NUM_USERS):
        uname = i
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)

        num_samples = len(X[i])
        train_len = int(0.75*num_samples)
        test_len = num_samples - train_len

        #X_train, X_test, y_train, y_test = train_test_split(X[i], y[i], train_size=0.75, stratify=y[i])\
        
        test_data['users'].append(uname)
        test_data["user_data"][uname] =  {'x': X[i][:test_len], 'y': y[i][:test_len]} 
        test_data['num_samples'].append(test_len)

        train_data["user_data"][uname] =  {'x': X[i][test_len:], 'y': y[i][test_len:]}
        train_data['users'].append(uname)
        train_data['num_samples'].append(train_len)
        
    return train_data['users'], train_data['user_data'], test_data['user_data']








"""def download_MovieLens_data():
    

    Description

    This script downloads MovieLens Datasets from GroupLens Research for the 
    matrix completion experiments. Before using the datasets, please read 
    the ters of use of GroupLens Research, included in the README files. 
    If the README files are missing or corrupted, you can find it at 
    "https://grouplens.org/datasets/movielens/" 
    Download the datasets and unzip (if the required files are missing)



    print('You are about to download MovieLens datasets from GroupLens Research Please read the terms of use in the README files first')

    dir_path = './data'

    # Check if the directory does not exist
    if not os.path.exists(dir_path):
    # Create the directory
        os.makedirs(dir_path)
        print(f' "{dir_path}" created successfully!')
    else:
        print(f'Folder "{dir_path}" already exists.')
    
    if ~exist('./data/ml-1m/ratings.dat','file')
    websave('data/ml-1m.zip','http://files.grouplens.org/datasets/movielens/ml-1m.zip')
    unzip('data/ml-1m.zip','data/')
    ~exist('./data/ml-100k/ub.train','file') || ~exist('./data/ml-100k/ub.test','file') 
    websave('data/ml-100k.zip','http://files.grouplens.org/datasets/movielens/ml-100k.zip')
    unzip('data/ml-100k.zip','data/')


    print('Data is downloaded and ready for use.\n');


def read_MovieLens_data():


"""


def read_data(dataset):
    '''parses data in given train and test data directories
    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    Return:
        clients: list of client ids
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    
    if(dataset == "CIFAR10"):
        clients, train_data, test_data = read_cifar10_data()
    
    elif(dataset == "CIFAR100"):
        clients, train_data, test_data = read_cifar100_data()

    elif(dataset == "MNIST"):
        clients, train_data, test_data = read_Mnist_data()
    
    elif(dataset == "FMNIST"):
        clients, train_data, test_data = read_FMnist_data()

    elif(dataset == "EMNIST"):
        clients, train_data, test_data = read_EMnist_data()

    else:
        print(" No dataset selected")
        
    
    
    return clients, train_data, test_data

    



def read_user_data(index,data,dataset):
    id = data[0][index]
    train_data = data[1][id]
    test_data = data[2][id]
    X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
    
    if(dataset == "CIFAR10"):
        X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS_CIFAR, IMAGE_SIZE_CIFAR, IMAGE_SIZE_CIFAR).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS_CIFAR, IMAGE_SIZE_CIFAR, IMAGE_SIZE_CIFAR).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
    
    elif(dataset == "CIFAR100"):
        X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS_CIFAR, IMAGE_SIZE_CIFAR, IMAGE_SIZE_CIFAR).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS_CIFAR, IMAGE_SIZE_CIFAR, IMAGE_SIZE_CIFAR).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)

    elif(dataset == "FMNIST"):
        X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
    
    elif(dataset == "MNIST"):
        X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)

    elif(dataset == "EMNIST"):
        X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)

    elif(dataset == "SYNTHETIC"):
        X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
    
    else:
        print("no dataset found")
    

    train_data = [(x, y) for x, y in zip(X_train, y_train)]
    test_data = [(x, y) for x, y in zip(X_test, y_test)]
    
    return train_data, test_data        



def select_users(users):
    selected_users = []
    for user in users:
        #print("user :",user)
        #input("press")
        participation = user.selection()
        if participation == 1 :
            selected_users.append(user)
        else:
            continue
    return selected_users
