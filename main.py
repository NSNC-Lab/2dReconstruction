"""Provides functions for finding an optimal two-dimensional reconstruction filter.

The reconstruction filter can be used to reconstruct an unknown one-dimensional signal (eg. signal envelope) from a known two-dimensional signal, such as multi-frequency neural spikes.
Can also be used to perform two-dimensional imputation. For a two dimensional signal with missing data, one can
train a reconstruction filter to fill in missing data.
"""

import torch.nn.functional as F
import torch

__author__ = "Junzi Dong"

__version__ = "1.0.0"
__maintainer__ = "Junzi Dong"
__email__ = "silakejd@gmail.com"
__status__ = "Prototype"

class TwoDimReconstruction(object):
    def __init__(self, n_runs, learning_rate=.0001, verbose=True):
        self.n_runs = n_runs
        self.learning_rate = learning_rate
        self.verbose = verbose

    def fit(self, spk, h, env):
        """
        Finds 2d reconstruction filter h using gradient descent. Must provide initial value for h.

        :param spk: pytorch tensor, shape (1, channels, time1). Spikes from which env will be reconstructed.
        :param h: pytorch tensor, requires gradient, shape (1, channels, time2). 2d reconstruction filter to fit.
        :param env: pytorch tensor, shape (time1, ). Envelope signal that spk and h are attempting to reconstruct.
        :return:
            h: optimized 2d reconstruction filter
            est_env: envelope estimated from h
            loss_history: loss value for gradient descent step
        """
        self.check_input(spk, h, env)
        n_output = spk.shape[2] - h.shape[2] + 1

        loss_history = []
        for t in range(self.n_runs):
            est_env = F.conv1d(spk, h)  # estimated envelope
            loss = (est_env - env[:n_output]).pow(2).sum()  # loss function
            loss_history.append(loss)  # append loss to loss history
            if self.verbose:
                print(f'Training loss: {loss}')
            loss.backward()  # back prop to calculate gradient of 2d reconstruction kernel
            with torch.no_grad():
                h -= self.learning_rate * h.grad
                h.grad.zero_()
        return h, est_env, loss_history

    @staticmethod
    def check_input(spk, h, env):
        """
        Make sure the dimensions of inputs are correct.

        :param spk: pytorch tensor, shape (1, channels, time1). Spikes from which env will be reconstructed.
        :param h: pytorch tensor, requires gradient, shape (1, channels, time2). 2d reconstruction filter to fit.
        :param env: pytorch tensor, shape (time1, ). Envelope signal that spk and h are attempting to reconstruct.
        :return: None
        """
        # dimensionality check
        if len(spk.shape) != 3:
            raise ValueError('Invalid spk shape, requires shape (1, channels, time)')
        else:
            if spk.shape[0] == 1 and spk.shape[1] >= 1 and spk.shape[2] >= 1:
                pass
            else:
                raise ValueError('Invalid spk shape, required shape: (1, channels, time)')

        if len(h.shape) != 3:
            raise ValueError('Invalid h shape, requires shape (1, channels, time)')
        else:
            if h.shape[0] == 1 and h.shape[1] >= 1 and h.shape[2] >= 1:
                pass
            else:
                raise ValueError('Invalid h shape, required shape: (1, channels, time)')

        if len(env.shape) != 1:
            raise ValueError('Invalid env shape, requires shape (time, )')
