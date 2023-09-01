import torch
import numpy as np

# Load data
Atrain = np.loadtxt('ml-100k/ub.base')
Atest = np.loadtxt('ml-100k/ub.test')

UserID = torch.tensor(Atrain[:, 0], dtype=torch.long)
MovID = torch.tensor(Atrain[:, 1], dtype=torch.long)
Rating = torch.tensor(Atrain[:, 2], dtype=torch.float)

UserID_test = torch.tensor(Atest[:, 0], dtype=torch.long)
MovID_test = torch.tensor(Atest[:, 1], dtype=torch.long)
Rating_test = torch.tensor(Atest[:, 2], dtype=torch.float)

# Clear all zero rows and columns
nU = UserID.max() + 1   # # Users
nM = MovID.max() + 1    # # Movies
nR = len(UserID)        # # Ratings

# Clear movies and users without rating in the train dataset
A = torch.sparse_coo_tensor((MovID, UserID), Rating, (nM, nU))
rDel = ~torch.sparse.sum(A, dim=1).bool().squeeze()
cDel = ~torch.sparse.sum(A, dim=0).bool().squeeze()
A = A[:, cDel][rDel, :]
MovID, UserID, Rating = A.coalesce().indices(), A.coalesce().values()

# Clear movies and users without rating in the test dataset
A = torch.sparse_coo_tensor((MovID_test, UserID_test), Rating_test, (nM, nU))
A = A[:, cDel][rDel, :]
MovID_test, UserID_test, Rating_test = A.coalesce().indices(), A.coalesce().values()

# Random permutation
p = torch.randperm(nR)
MovID, UserID, Rating = MovID[p], UserID[p], Rating[p]
MovID, UserID, Rating = MovID[:nR], UserID[:nR], Rating[:nR]

# Split data into clients
client_num = 10                  # # Clients
client_data = nR // client_num   # # Clients' data

MovID_client = torch.zeros((client_data, client_num), dtype=torch.long)
UserID_client = torch.zeros((client_data, client_num), dtype=torch.long)
Rating_client = torch.zeros((client_data, client_num), dtype=torch.float)

for i in range(client_num):
    MovID_client[:, i] = MovID[i*client_data: (i+1)*client_data]
    UserID_client[:, i] = UserID[i*client_data: (i+1)*client_data]
    Rating_client[:, i] = Rating[i*client_data: (i+1)*client_data]

alpha = 7000

Data_ml100k_Nuclear = {}
Data_ml100k_Nuclear['client_num'] = client_num
Data_ml100k_Nuclear['user_num'] = nU
Data_ml100k_Nuclear['movie_num'] = nM
Data_ml100k_Nuclear['rating_num'] = nR
Data_ml100k_Nuclear['MovID'] = MovID
Data_ml100k_Nuclear['UserID'] = UserID
Data_ml100k_Nuclear['Rating'] = Rating
Data_ml100k_Nuclear['MovID_client'] = MovID_client
Data_ml100k_Nuclear['UserID_client'] = UserID_client
Data_ml100k_Nuclear['Rating_client'] = Rating_client
Data_ml100k_Nuclear['MovID_test'] = MovID_test
Data_ml100k_Nuclear['UserID_test'] = UserID_test
Data_ml100k_Nuclear['Rating_test'] = Rating_test









###################################### Code not working #####################################

# Define the dataset class
def read_MovieLens_data(dataset_name):
    """
    data[0] = train data
    data[1] = test data
    
    """
    train_data, test_data = MovieLensDataset(dataset_name)
    print(train_data[0][1])
    
    UserID_train = torch.tensor([x[0] for x in train_data])
    MovID_train = torch.tensor([x[1] for x in train_data])
    Rating_train = torch.tensor([x[2] for x in train_data])

    UserID_test = torch.tensor([x[0] for x in test_data] )
    MovID_test = torch.tensor([x[1] for x in test_data] )
    Rating_test = torch.tensor([x[2] for x in test_data])

    ### clear all zero rows and columns

    nU = UserID_train.max() +1 # users
    nM = MovID_train.max() +1 # Movies
    nR = len(UserID_train) # Rating
    
    print("nU :",nU,"nM :",nM,"nR :",nR)
    # Clear movies and users without rating in the train dataset
    # print("UserID_train :",len(UserID_train))
    # print("UserID_train :",len(MovID_train))
    # print("UserID_train :",len(Rating_train))

    # print(UserID_train)
    indices = torch.stack([UserID_train, MovID_train], dim=0)     # print(list_tensor)
    # print(list_tensor)
    A = torch.sparse_coo_tensor(indices, Rating_train, (nU, nM))
    dense_A = A.to_dense()
    rSum_A = dense_A.sum(dim=1)
    cSum_A = dense_A.sum(dim=0)
    rDel = ~(rSum_A.bool().squeeze())
    cDel = ~(cSum_A.bool().squeeze())
    
    print(len(rDel))
    print(len(cDel))
    A = A.coalesce() 
    A = A[:, cDel]
    A = A[rDel, :]
    # apply boolean and squeeze operations to compute rDel
    # rDel = ~(rSum_A.bool().squeeze())
    # cDel = ~(cSum_A.bool().squeeze())
    # rDel = ~(A.to_dense().any(dim=1))
    # cDel = ~(A.to_dense().any(dim=0))

    # A = A.to_dense()[rDel, :]
    #A = A[:, cDel]
    
    #UserID_train, MovID_train = torch.nonzero(A) 
    
    # remove rows and columns with all zeros
    # rDel = ~(A.coalesce().to_dense().sum(dim=1).bool())
    # cDel = ~(A.coalesce().to_dense().sum(dim=0).bool())
    # A = A.coalesce().indices()
    # A = A[:, torch.logical_and(rDel, cDel)]

    # find non-zero elements
    # MovID, UserID = A[0], A[1]

# print MovID and UserID
    # print(MovID)
    # print(UserID)   
    
    
    
    # MovID_train, UserID_train, Rating_train = A.coalesce().indices(), A.coalesce().values()

    # Clear movies and users without rating in the test dataset
   # A = torch.sparse_coo_tensor((MovID_test, UserID_test), Rating_test, (nM, nU))
  #  A = A[:, cDel][rDel, :]
  #  MovID_test, UserID_test, Rating_test = A.coalesce().indices(), A.coalesce().values()

    # Random permutation
    # p = torch.randperm(nR)
    # MovID_train, UserID_train, Rating_train = MovID[p], UserID[p], Rating[p]
    # MovID, UserID, Rating = MovID[:nR], UserID[:nR], Rating[:nR]






    
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)








