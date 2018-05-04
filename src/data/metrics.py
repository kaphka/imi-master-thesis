import numpy as np

def mean_gradient(bw):
    dx = bw[1:, :] - bw[:-1, :]
    dy = bw[:, 1:] - bw[:, :-1]
    grad = np.add(np.power(dx[:, :-1], 2), np.power(dy[:-1, :], 2))
    return np.mean(grad)