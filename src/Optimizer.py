import torch
from torch.optim import Optimizer, sgd
from src.client import FedFW_client
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
            num_client_iter=num_client_iter,
            step_direction_func=step_direction_func,
            alpha=alpha
            )
        
        self.server_model = copy.deepcopy(server_model)
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
                # print(alpha)
                # input("press")
                num_client_iter = group['num_client_iter']

                # Compute eta_t and lambda_t
                # print(step)
                eta_t = 2 / (step + 1)
                # print("eta_t :", eta_t)
                lambda_t = lambda_0 * math.sqrt(step + 1)
                # print("lambda_t :", lambda_t)
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















class FW(Optimizer):



    def __init__(self, params, eta_t, kappa, device):
        
        if eta_t <0:
            raise ValueError("Invalid Learning rate: {}".format(eta_t))
        defaults = dict(eta_t=eta_t)
        super(FW, self).__init__(params, defaults)
        self.eta_t=eta_t
        self.kappa=kappa
        self.device=device
    def FW_LMO(self, g_it):
        all_param = [] 
        param_size = []
        for j, param in enumerate(g_it.parameters()):
          
            all_param.append(param.data.view(-1))
            param_size.append(len(param.data.view(-1)))
          
        param_concat = torch.cat(all_param, dim=0)
        lmo_s_it = LMO_l1(param_concat.cpu().numpy(), self.kappa)
        lmo_s_it = torch.from_numpy(lmo_s_it)

        param_proj = torch.split(lmo_s_it, param_size, dim=0)
        param_size_now = []
        for proj , param in zip(param_proj, g_it.parameters()):
            parameter_size = param.shape
            param.data = proj.view(*parameter_size)
        g_it.to(self.device)
        return g_it.parameters()
    
    def step(self, g_it, x_it, x_bar_t, n, lambda_t, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # for group in self.param_groups:
        #    for g_it_param, x_it_param, x_bar_t_param in zip(g_it.parameters(), group['params'], x_bar_t.parameters()):
                #print("g_it_param.data :", g_it_param.data)
                #input("press") 
                #print("x_it_param.grad.data :", x_it_param.grad.data)
                #input("press") 
        #        print("x_bar_t_param.data :", x_bar_t_param.data)
        #        input("press") 
        #        g_it_param.data = x_it_param.grad.data*(1/n) + lambda_t*(x_it_param.data -x_bar_t_param.data)
        #        print("g_it_param.data :", g_it_param.data)    
        #        input("press")    
        
        for g_it_param, x_it_param, x_bar_t_param in zip(g_it.parameters(), x_it.parameters(), x_bar_t.parameters()):
            g_it_param.data = x_it_param.grad.data*(1/n) + lambda_t*(x_it_param.data -x_bar_t_param.data)
        s_it = self.LMO_l1(g_it)
        # print("s_it :", list(s_it))
        # input("press")
        
        for x_it_param, s_it_param in zip(x_it.parameters(), s_it):
            x_it_param.data = (1-self.eta_t)*x_it_param.data + self.eta_t*s_it_param.data
        # print(group['params'])
        # input("press")
        
        
        # for x_it_param in x_it.parameters():
        #        x_it_param.data = (1-self.eta_t)*x_it_param.data + self.eta_t*x_it_param.grad.data
        
        
        return x_it.parameters()
                

        for group in self.param_groups:
            kappa = group['kappa']

            for p in group['params']:
                if p.grad is None:
                    continue
                s = LMO_l1(p.grad.data.numpy(), kappa)
                gamma = 2 / (self.k + 2)
                # x^(k+1) = x^(k) - g x^(k) + g s
                delta_p = torch.Tensor(gamma * s - gamma * p.data.numpy())
                p.data.add_(delta_p)
        
        self.k += 1
        return loss
