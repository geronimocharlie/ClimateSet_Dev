import torch
from torch import Tensor
import numpy as np
from scipy.linalg import expm
from typing import List, Union


class Gumbel_Sigmoid(torch.nn.Module):

    """
    Adapted from https://github.com/AngelosNal/PyTorch-Gumbel-Sigmoid.git

    Samples from the Gumbel-Sigmoid distribution and optionally discretizes.
    The discretization converts the values greater than `threshold` to 1 and the rest to 0.
    The code is adapted from the official PyTorch implementation of gumbel_softmax:
    https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized,
            but will be differentiated as if it is the soft sample in autograd
     threshold: threshold for the discretization,
                values greater than this will be set to 1 and the rest to 0

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Sigmoid distribution.
      If ``hard=True``, the returned samples are descretized according to `threshold`, otherwise they will
      be probability distributions.

    """

    def __init__(self, tau: float = 1, hard: bool = False, threshold: float = 0.5):
        super(Gumbel_Sigmoid, self).__init__()
        self.tau=tau
        self.hard=hard
        self.threshold=threshold

    def forward(self, logits: Tensor) -> Tensor:
        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )  # ~Gumbel(0, 1)
        gumbels = (logits + gumbels) / self.tau  # ~Gumbel(logits, tau)
        y_soft = gumbels.sigmoid()

        if self.hard:
            # Straight through.
            indices = (y_soft > self.threshold).nonzero(as_tuple=True)
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
            y_hard[indices[0], indices[1]] = 1.0
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret

class ALM:
    """
    Adapted from: https://github.com/RolnickLab/causalpaca/blob/causal_model/causal/utils.py (internal)
    
    Augmented Lagrangian Method
    To use the quadratic penalty method (e.g. for the acyclicity constraint),
    just ignore 'self.lambda'
    """
    def __init__(self,
                 mu_init: float,
                 mu_mult_factor: float,
                 omega_gamma: float,
                 omega_mu: float,
                 h_threshold: float,
                 min_iter_convergence: int,
                 dim_gamma = [1]):
        self.gamma = torch.zeros(*dim_gamma)
        self.delta_gamma = -np.inf
        self.mu = mu_init
        self.min_iter_convergence = min_iter_convergence
        self.h_threshold = h_threshold
        self.omega_mu = omega_mu
        self.omega_gamma = omega_gamma
        self.mu_mult_factor = mu_mult_factor
        self.stop_crit_window = 100
        self.constraint_violation = []
        self.has_converged = False
        self.dim_gamma = dim_gamma

    def _compute_delta_gamma(self, iteration: int, val_loss: list):
        # compute delta for gamma
        if iteration >= 2 * self.stop_crit_window and \
           iteration % (2 * self.stop_crit_window) == 0:
            t0, t_half, t1 = val_loss[-3], val_loss[-2], val_loss[-1]

            # if the validation loss went up and down, do not update lagrangian and penalty coefficients.
            if not (min(t0, t1) < t_half < max(t0, t1)):
                self.delta_gamma = -np.inf
            else:
                self.delta_gamma = (t1 - t0) / self.stop_crit_window
        else:
            self.delta_gamma = -np.inf  # do not update gamma nor mu

    def update(self, iteration: int, h_list: list, val_loss: list):
        """
        Update the value of mu and gamma. Return True if it has converged.
        Args:
            iteration: number of training iterations completed
            h_list: list of the values of the constraint
            val_loss: list of validation loss
        """
        self.has_increased_mu = False

        if len(val_loss) >= 3:
            h = h_list[-1]
            if len(self.dim_gamma) > 1:
                h_scalar = torch.sum(h)
            else:
                h_scalar = h

            # check if QPM has converged
            if iteration > self.min_iter_convergence and h_scalar <= self.h_threshold:
                self.has_converged = True
            else:
                # update delta_gamma
                self._compute_delta_gamma(iteration, val_loss)

                # if we have found a stationary point of the augmented loss
                if abs(self.delta_gamma) < self.omega_gamma or self.delta_gamma > 0:
                    self.gamma += self.mu * h
                    self.constraint_violation.append(h_scalar)

                    # increase mu if the constraint has sufficiently decreased
                    # since the last subproblem
                    if len(self.constraint_violation) >= 2:
                        if h_scalar > self.omega_mu * self.constraint_violation[-2]:
                            self.mu *= self.mu_mult_factor
                            self.has_increased_mu = True


class TrExpScipy(torch.autograd.Function):
    """

    adapted from: https://github.com/RolnickLab/causalpaca/blob/causal_model/causal/dag_optim.py
    autograd.Function to compute trace of an exponential of a matrix
    """

    @staticmethod
    def forward(ctx, input):
        with torch.no_grad():
            # send tensor to cpu in numpy format and compute expm using scipy
            expm_input = expm(input.detach().cpu().numpy())
            # transform back into a tensor
            expm_input = torch.as_tensor(expm_input)
            if input.is_cuda:
                # expm_input = expm_input.cuda()
                assert expm_input.is_cuda
            # save expm_input to use in backward
            ctx.save_for_backward(expm_input)

            # return the trace
            return torch.trace(expm_input)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            (expm_input,) = ctx.saved_tensors
            return expm_input.t() * grad_output



def init_weigths(m: torch.nn.Linear, value: float = 5.0):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(value)

class WeightClipper(object):
    def __init__(self, min: float = 0.0, max: float= np.inf):
        self.min = min
        self.max = max
    
    def __call__(self, module: torch.nn.Module):
        # filter the variables to get the ones you want

        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(self.min,self.max)
        
"""
DCDDI repo below


def sample_logistic(shape, uniform):
    u = uniform.sample(shape)
    return torch.log(u) - torch.log(1 - u)

def gumbel_sigmoid(log_alpha, uniform, bs, tau=1, hard=False):
    shape = tuple([bs] + list(log_alpha.size()))
    logistic_noise = sample_logistic(shape, uniform)

    y_soft = torch.sigmoid((log_alpha + logistic_noise) / tau)

    if hard:
        y_hard = (y_soft > 0.5).type(torch.Tensor)

        # This weird line does two things:
        #   1) at forward, we get a hard sample.
        #   2) at backward, we differentiate the gumbel sigmoid
        y = y_hard.detach() - y_soft.detach() + y_soft

    else:
        y = y_soft

    return y
"""
