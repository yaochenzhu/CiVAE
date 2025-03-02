import torch
import torch.nn as nn

from pyro.distributions.torch_distribution import TorchDistributionMixin
from pyro.distributions.util import broadcast_shape, sum_rightmost
from pyro.ops.special import log_binomial

from torch.autograd import grad


class UnnormalizedExponentialFamily(TorchDistributionMixin):

    def __init__(self, t_f, t_nf, lambd_f, lambd_nf, *args, **kwargs):
        self.t_f = t_f
        self.t_nf = t_nf
        self.lambd_f = lambd_f
        self.lambd_nf = lambd_nf

        ### Infer the batch shape and event shape
        self.batch_shape = torch.Size([self.lambd_f.shape[0]])
        self.event_shape = torch.Size([self.lambd_f.shape[1]//2])
        return super().__init__(*args, **kwargs)

    def log_prob(self, value):
        ### Used for score matching purpose
        ### Calculate the negative match score
        ### which will be optimized normally in the Elbo
        z = value
        n_samples = z.shape[0]

        ### Calculate the sufficient statistics
        t_f = self.t_f(z)
        t_nf = self.t_nf(z)
        t = torch.concat([t_f, t_nf], axis=-1)

        ### Calculate the parameters
        lambd = torch.concat([self.lambd_f, self.lambd_nf], axis=-1)

        ### Calculate the unnormalized log-density corresponds to 
        ### the nonfactorized part
        pnf_hat = torch.mul(t_nf, self.lambd_nf).sum(axis=1, keepdims=True)

        ### Calculate the negative match score
        neg_match = torch.zeros((n_samples))

        ### Calculate the factorized part
        neg_match -= 2*self.lambd_f[:, self.event_shape[0]:].sum(axis=-1)
        neg_match -= 0.5*torch.pow(self.lambd_f[:, :self.event_shape[0]], 2).sum(axis=-1)

        ### Calculate the non-factorized part
        for i in range(n_samples):
            grads = grad(pnf_hat[i], z, create_graph=True, retain_graph=True)[0][i]
            #neg_match[i] -= 0.5 * torch.pow(grads, 2).sum() * 0.1
            neg_match[i] -= 0.5 * torch.pow(grads, 2).sum()

        return neg_match

    
    def sample(self):
        pass
