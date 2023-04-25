
from src.utils import Gradient_nuclear
import numpy as np


def RMSE_nuclear(x, nM, nU, MovID, UserID, Rating):
    # RMSE
    ind_train = np.ravel_multi_index((MovID-1, UserID-1), dims=(nM, nU))
    err = np.sqrt(np.mean((x[ind_train] - Rating)**2))
    return err


def gradient_descent(x, eta, grad):
    x = x - eta * grad

    return x

def matrix_comp(data):


    num_clients = data['client_num']
    print("num_client :",num_clients)
    user_num = data['user_num']
    movie_num = data['movie_num']
    rating_num = data['rating_num'] 
    MovID =  data['MovID']
    UserID = data['UserID']
    Rating = data['Rating']
    MovID_client = data['MovID_client'].T
    
    # print("MovID_client :", len(MovID_client))
    # print("MovID_client :",len(MovID_client[0]))
    
    UserID_client = data["UserID_client"]
    UserID_client = UserID_client.T
    # print("UserID_client :",len(UserID_client))
    Rating_client = data["Rating_client"].T
    print("rating client :",len(Rating_client[0]))
    MovID_test = data["MovID_test"]
    UserID_test = data["UserID_test"]
    Rating_test = data["Rating_test"]
    alpha = data["alpha"]
    Data_name = data["Data_name"]
    
    x = np.zeros((user_num, movie_num))  # 3 rows, 4 columns
    x_bar = x 
    global_iters = 10
    lr = 0.1
    all_grad = []
 
    for i in range(0, num_clients):
        all_x.append(x)

    for glob_iter in range(0, global_iters):
        print("------ Global iteration [",glob_iter,"]")
        all_x = []
        #### Client update
        for i in range(0,num_clients):
            grad = Gradient_nuclear.grad_nuclear(x_bar, user_num, movie_num, MovID_client[i], UserID_client[i], Rating_client[i])
            x = gradient_descent(all_x[i], lr, grad)
            
            all_x.append(x)

        ## server update 
        x_bar = avg(all_x)
        



    # return 0