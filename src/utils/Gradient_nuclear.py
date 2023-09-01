from scipy.sparse import coo_matrix
import numpy as np

def grad_nuclear(x, nM, nU, MovID, UserID, Rating):
    ind_train = np.ravel_multi_index((MovID, UserID), dims=(nM, nU))
    print(ind_train)
    gf_forw = x[ind_train] - Rating
    I, J = np.unravel_index(ind_train, dims=(nM, nU))
    gradf = coo_matrix((gf_forw, (I, J)), shape=(nM, nU)).toarray()
    return gradf


