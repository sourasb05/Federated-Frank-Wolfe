import numpy as np 
# from projections import euclidean_proj_l1ball


def LMO_l1(grad, kappa):
    shape = grad.shape
    grad = grad.reshape(-1)
    s = np.zeros(grad.shape)
    coord = np.argmax(np.abs(grad))
    s[coord] = kappa * np.sign(grad[coord])
    return - s.reshape(*shape)

def LMO_l2(grad, kappa):
    shape = grad.shape
    grad = grad.reshape(-1)
    l2_norm = np.linalg.norm(grad) 
    # print("l2_norm :",l2_norm)
    grad = (grad / l2_norm)*kappa
    #print(grad)
    #input("press")
    return grad

"""def P_l1(grad, kappa):
    shape = grad.shape
    grad = grad.reshape(-1)
    proj = euclidean_proj_l1ball(grad, kappa)
    return proj.reshape(*shape)
"""