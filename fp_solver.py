""" A fixed point solver
"""
from abc import abstractmethod
import torch
from torch import autograd


class FixedPointSolver(object):
    """ fixed point solver base class """
    @abstractmethod
    def get_fixed_point(self, init_states, energy_fn):
        """
        :param init_states: A list of tensor
        :param energy_fn: A function that take `states` and return energy for each example
        :return: The fixed point state
        """
        pass


class FixedStepSolver(FixedPointSolver):
    """ Use step size each time """
    def __init__(self, step_size, max_steps=500):
        self.step_size = step_size
        self.max_steps = max_steps

    def get_fixed_point(self, states, energy_fn):
        """ Use fixed step size gradient decsent """
        step = 0
        while step < self.max_steps:
            energy = energy_fn(states)
            grads = autograd.grad(-torch.sum(energy), states)
            for tensor, grad in zip(states, grads):
                tensor[:] = tensor + self.step_size * grad
                tensor[:] = torch.clamp(tensor, 0, 1)
            step += 1
        return states


class MaxGradNormSolver(FixedPointSolver):
    """ Use step size each time """
    def __init__(self, step_size, max_grad_norm=1e-6, max_steps=500):
        self.step_size = step_size
        self.max_steps = max_steps
        self.max_grad_norm = max_grad_norm
        self._logs = dict()

    def get_fixed_point(self, states, energy_fn):
        """ Use fixed step size gradient decsent """
        for step in range(self.max_steps):
            energy = energy_fn(states)
            grads = autograd.grad(-torch.sum(energy), states)
            for tensor, grad in zip(states, grads):
                tensor[:] = tensor + self.step_size * grad
                tensor[:] = torch.clamp(tensor, 0, 1)

            grad_norm = self._get_grad_norm(grads)
            if grad_norm < self.max_grad_norm:
                break
        self._logs['steps_made'] = step
        self._logs['grad_norm'] = grad_norm

        return states

    @staticmethod
    def _get_grad_norm(grads):
        return sum(grad.norm() for grad in grads)
