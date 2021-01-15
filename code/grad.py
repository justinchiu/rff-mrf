import math 
import torch
from torch.autograd import grad

# probability space computations
def phi(X, W):
    return torch.exp(X @ W - X.pow(2).sum(-1, keepdim=True) / 2)

def rff(X, Y, W):
    A, B = phi(X, W), phi(Y, W)
    values = A @ B.T
    return values / values.sum(-1, keepdim=True)

# log space computations
def log_phi(X, W):
    return X @ W - X.pow(2).sum(-1, keepdim=True) / 2

def logbmm(A, B):
    C = A[:,None] + B[None]
    return C.logsumexp(-1)

def log_rff(X, Y, W):
    A, B = log_phi(X, W), log_phi(Y, W)
    values = logbmm(A, B)
    return values.log_softmax(-1)

# shift by max
def shift_log_phi(X, W):
    A = X @ W - X.pow(2).sum(-1, keepdim=True) / 2
    return A - A.max(-1, keepdim=True)[0]

def shift_log_rff(X, Y, W):
    A, B = shift_log_phi(X, W), log_phi(Y, W)
    B = B - B.max()
    values = logbmm(A, B)
    return values.log_softmax(-1)

# detach and shift max
def detach_shift_log_phi(X, W):
    d = W.shape[0]
    ratio = 1 / math.sqrt(d)
    A = math.log(ratio) + X @ W - X.pow(2).sum(-1, keepdim=True) / 2
    return A - A.max(-1, keepdim=True)[0].detach()

def detach_shift_log_rff(X, Y, W):
    A, B = detach_shift_log_phi(X, W), log_phi(Y, W)
    B = B - B.max().detach()
    values = logbmm(A, B)
    return values.log_softmax(-1)


n = 32
num_features = 128
d = 64

T = 1

X = torch.randn(n, d) / T
Y = torch.randn(n, d) / T
W = torch.randn(d, num_features) / T

X.requires_grad = True

out1 = rff(X, Y, W)
out2 = log_rff(X, Y, W).exp()
out3 = shift_log_rff(X, Y, W).exp()
out4 = detach_shift_log_rff(X, Y, W).exp()

grad1, = grad(out1.sum(), X)
grad2, = grad(out2.sum(), X)
grad3, = grad(out3.sum(), X)
grad4, = grad(out4.sum(), X)

