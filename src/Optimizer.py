import torch
from torch.optim import Optimizer, sgd
from src.utils.oracles import LMO_l1


class FW(Optimizer):
    """Stochastic Frank Wolfe with |vec(W_i)|_1 <= kappa_l1 where W_i are parameter sets
    """
    def __init__(self, params, kappa):
        self.k = 0
        assert kappa > 0
        defaults = dict(kappa=kappa)
        super(FW, self).__init__(params, defaults)

    def step(self, g_it, x_it, x_bar_t, n, lambda_t, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for g_it_param, x_it_param , x_bar_t_param in zip(g_it.parameters(), x_it.parameters(), x_bar_t.parameters()):
            g_it_param.data = x_it_param.grad.data*(1/n) + lambda_t*(x_it_param.data -x_bar_t_param.data)
        
        return g_it
                

    """   for group in self.param_groups:
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
    """