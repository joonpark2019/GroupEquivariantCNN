import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def weight(rho):
    return F.sigmoid(rho)


def loss(xy, rho, N, M):
    i = np.linspace(0, N, N)
    j = np.linspace(0, M, M)
    x, y = np.meshgrid(i, j)
    print(x)
    print(y)
