import time
import numpy as np


class MovingAverage():
    def __init__(self, inertia=0.9):
        self.avg = 0.
        self.inertia = inertia
        self.reset()

    def reset(self):
        pass

    def update(self, val):
        self.avg = self.inertia * self.avg + (1 - self.inertia) * val


def cpu_sk(S, lambda_=1):
    """ Sinkhorn Knopp optimization on CPU
        * stores activations to RAM
        * does matrix-vector multiplies on CPU
        * slower than GPU
    """
    # 1. aggregate inputs:
    N = S.shape[0]
    K = S.shape[1]
    if K == 1:
        return S

    # 2. solve label assignment via sinkhorn-knopp:
    S_posterior = optimize_S_sk(S, K, N, lambda_)
    return S_posterior


def optimize_S_sk(S, K, N, lambda_):
    tt = time.time()
    S_posterior = np.copy(S).T  # now it is K x N
    r = np.ones((K, 1)) / K
    c = np.ones((N, 1)) / N
    S_posterior **= lambda_  # K x N
    inv_K = 1. / K
    inv_N = 1. / N
    err = 1e6
    _counter = 0
    while err > 1e-1:
        r = inv_K / (S_posterior @ c)  # (KxN)@(N,1) = K x 1
        c_new = inv_N / (r.T @ S_posterior).T  # ((1,K)@(KxN)).t() = N x 1
        if _counter % 10 == 0:
            err = np.nansum(np.abs(c / c_new - 1))
        c = c_new
        _counter += 1

    # inplace calculations.
    S_posterior = S_posterior.T
    S_posterior *= c * N
    S_posterior = S_posterior.T
    S_posterior *= r
    S_posterior = S_posterior.T

    return S_posterior
