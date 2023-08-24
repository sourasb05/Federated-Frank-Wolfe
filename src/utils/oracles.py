import numpy as np 
import torch
# from FLAlgorithms.optimizers.simplex_projection import euclidean_proj_l1ball

# from projections import euclidean_proj_l1ball


def LMO_l1(grad, kappa):
    shape = grad.shape
    grad = grad.reshape(-1)
    s = torch.zeros(grad.shape)
    coord = torch.argmax(torch.abs(grad))
    s[coord] = kappa * torch.sign(grad[coord])
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


def LMO_l1(grad, alpha):
    shape = grad.shape
    grad = grad.reshape(-1)
    s = torch.zeros_like(grad.data)
    coord = torch.argmax(torch.abs(grad))
    s[coord] = alpha * torch.sign(grad[coord])
    return - s.reshape(*shape)


"""def Projection_l1(grad, alpha):
    shape = grad.shape
    grad = grad.reshape(-1)
    proj = euclidean_proj_l1ball(grad, alpha)
    return proj.reshape(*shape)

"""