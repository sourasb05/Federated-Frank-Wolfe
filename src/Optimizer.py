import torch
from torch.optim import Optimizer
from src.utils.oracles import LMO_l1 , LMO_l2
from typing import Callable
import copy
import math



class FedFW(Optimizer):
    """ Implements the FedFW algorithm for a general step direction function

    Parameters:
        server_model (nn.Module): server model parameters
        lambda_0 (float): starting value of Frank-Wolfe penalty parameter, should be >= 0
        num_client_iter (int): total number of client iterations, should be > 0
        step_direction_func (func): a function that takes the gradient tensor and returns the step direction.
        alpha (float): Threshold for constraint set, should be > 0
    """

    def __init__(self,
                    params, 
                    server_model, 
                    lambda_0: float, 
                    eta_t: float,
                    eta_type: str,
                    lambda_type: str,
                    num_client_iter: int,
                    step_direction_func: Callable[[torch.Tensor, float], torch.Tensor], 
                    alpha: float):
        
        if not 0.0 <= lambda_0:
            raise ValueError("Invalid starting Frank-Wolfe penalty parameter lambda {} - should be in >= 0".format(lambda_0))
        if not 0 < num_client_iter:
            raise ValueError("Invalid total number of client iterations {} - should be in > 0".format(num_client_iter))
        if not 0.0 < alpha:
            raise ValueError("Invalid threshold for constraint set {} - should be in >= 0".format(alpha))

        defaults = dict(
            lambda_0=lambda_0,
            eta_t=eta_t,
            num_client_iter=num_client_iter,
            step_direction_func=step_direction_func,
            alpha=alpha
            )
        
        self.server_model = copy.deepcopy(server_model)
        self.eta_type = eta_type
        self.lambda_type = lambda_type
        super(FedFW, self).__init__(params, defaults)


    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            for (server_p, p) in zip(self.server_model.parameters(), group['params']):
                if p.grad is None:
                    continue
                # grad = p.grad.data
                # if grad.is_sparse:
                #    raise RuntimeError('FedFW does not support sparse gradients')
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 1
                    state['step_direction'] = torch.zeros_like(p.data)
                    state['eta_t'] = 1

                step = state['step']

                lambda_0 = group['lambda_0']
                step_direction_func = group['step_direction_func']
                alpha = group['alpha']
                
                num_client_iter = group['num_client_iter']

                # Compute eta_t and lambda_t
                # print(step)
                # eta_t = 2 / (step + 1)
                if self.eta_type == "constant":
                    eta_t = group['eta_t'] ## 
                else:
                    eta_t = 2 / (step + 1)

                if self.lambda_type == "constant":
                    lambda_t = group['lambda_0']
                else:
                    lambda_t = lambda_0 * math.sqrt(step + 1)

                # Compute g_i^t
                # grad.mul_(1 / num_client_iter).add_(p.data - server_p.data, alpha=lambda_t)
                grad = (1/ 10)*p.grad.data + lambda_t*(p.data - server_p.data)
                # Compute step direction from g_i^t
                fw_step_direction = step_direction_func(grad, alpha)
                # print(fw_step_direction)
                # input("press")
                # x_i^{t + 1} = (1 - eta_t)*x_i^t + eta_t*s_i^t
                p.data.mul_(1 - eta_t).add_(fw_step_direction, alpha=eta_t)
                # p.data.mul_(1 - eta_t).add_(grad, alpha=eta_t)
                # p.data = (1 -  eta_t)*p.data + eta_t*grad
               #  p.data.mul_(1 - eta_t).add_(grad, alpha=eta_t)
                state['step'] += 1
                # state["step_direction"] = fw_step_direction
                state["eta_t"] = eta_t
        return loss



class MySGD(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(MySGD, self).__init__(params, defaults)

    def step(self, closure=None, beta = 0):
        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            # print(group)
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data = p.data - (group['lr'] * d_p)
        
        return loss
