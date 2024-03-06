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
                    lambda_0: float, 
                    eta_t: float,
                    eta_type: str,
                    lambda_type: str,
                    num_client_iter: int,
                    step_direction_func: Callable[[torch.Tensor, float], torch.Tensor], 
                    kappa: float,
                    algorithm: str):
        
        if  lambda_0 <= 0.0:
            raise ValueError("Invalid starting Frank-Wolfe penalty parameter lambda {} - should be in >= 0".format(lambda_0))
        if num_client_iter <= 0.0:
            raise ValueError("Invalid total number of client iterations {} - should be in > 0".format(num_client_iter))
        if kappa <= 0.0:
            raise ValueError("Invalid threshold for constraint set {} - should be in >= 0".forma(kappa))

        defaults = dict(
            lambda_0=lambda_0,
            eta_t=eta_t,
            num_client_iter=num_client_iter,
            step_direction_func=step_direction_func,
            kappa=kappa
            )
        
        
        self.eta_type = eta_type
        self.lambda_type = lambda_type
        self.algorithm = algorithm
        super(FedFW, self).__init__(params, defaults)


    def step(self, s_it, server_model, y_it, closure=None):
        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            for (server_p, p, s, y_it_p) in zip(server_model.parameters(), group['params'], s_it.parameters(), y_it.parameters()):
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
                kappa = group['kappa']
                
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
                if self.algorithm == "FedFW_plus":
                    y_it_p.data += lambda_0*(p.data - server_p.data)
                    grad = (1/ 10)*p.grad.data + lambda_t*(p.data - server_p.data) + y_it_p.data
                else:
                    grad = (1/ 10)*p.grad.data + lambda_t*(p.data - server_p.data)
                # Compute step direction from g_i^t
                
                fw_step_direction = step_direction_func(grad, kappa)
                # print("101 :", fw_step_direction)

               
                s.data = fw_step_direction.clone()
          
                p.data.mul_(1 - eta_t).add_(fw_step_direction, alpha=eta_t)
                p.grad.data = grad
                state['step'] += 1
                state["eta_t"] = eta_t
        
        # for s in s_it.parameters():
        #    print(s.data)
        # input("press")
        l_s_it = copy.deepcopy(list(s_it.parameters()))
        return loss , l_s_it



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



class PerturbedSGD(Optimizer):
    """Perturbed SGD optimizer"""

    def __init__(self, params, lr=0.01, alpha=0.0):

        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, alpha=alpha)
        self.idx = 0

        super(PerturbedSGD, self).__init__(params, defaults)
    
    def step(self, y_ik, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure

        yik_update = y_ik.parameters()
        # internal sgd update
        for group in self.param_groups:
            #get the lr
            lr = group['lr']
            alpha = group['alpha']

            for param, y_param in zip(group['params'], yik_update):
                param.grad.data = param.grad.data + alpha*(param.data - y_param.data)
                param.data = param.data -lr*param.grad.data
                
       
        return group['params'], loss
    

    """ 
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['v_star'] = torch.zeros_like(p)
                
    def update_v_star(self, v_star):

        current_index = 0

        for group in self.param_groups:
            for p in group['params']:
                numel = p.data.numel()
                size = p.data.size()

                state = self.state[p]
                state['v_star'] = (v_star[current_index:current_index+numel].view(size)).clone()

                current_index += numel

    @torch.no_grad()
""" 
