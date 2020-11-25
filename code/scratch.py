import numpy as np

x = np.random.randn(3, 5)
y = np.random.randn(3, 5)
z = np.random.randn(3, 5)

# linear approx linear chain crf correctness
# elimination order: z -> x -> x
Zs = x.dot(y.T.dot(y.dot(z.sum(0)))).sum()
Zp = x.sum(0).dot(y.T.dot(y)).dot(z.sum(0))

# breaks with batching
inner = lambda x,y: np.tensordot(x,y,1)
outer = lambda x,y: np.tensordot(x,y,0)

# example of tensor version for distributive rule
a = np.random.randn(5)
b = np.random.randn(5)
c = np.random.randn(5)
meh = a.dot(b) * a.dot(c)
A = np.einsum("i,j->ij", a, a)
out = np.tensordot(np.tensordot(A, b, axes=1), c, axes=1)
out2 = inner(inner(outer(a, a), b), c)

x = np.random.randn(3, 5)
y = np.random.randn(3, 5)
z = np.random.randn(3, 5)
w = np.random.randn(3, 5)

# S_x x^T S_y y (S_z y^T z) (S_w y^T w)
naive = np.einsum("xi,yi,yj,zj,yk,wk->",x,y,y,z,y,w)
naive2 = x.dot((y * y.dot(z.sum(0))[:,None] * y.dot(w.sum(0))[:,None]).sum(0)).sum()

# do sums in parallel
Y = np.einsum("yi,yj,yk->ijk",y,y,y)
X = x.sum(0)
Z = z.sum(0)
W = w.sum(0)
yxzw = np.einsum("ijk,i,j,k->",Y,X,Z,W)

# broken with batching
Yo = outer(outer(y,y), y)

"""
x = np.random.randn(1, 5)
y = np.random.randn(1, 5)
z = np.random.randn(1, 5)
# main computation is x (x^Ty) (x^Tz), want to pull out x?
X = (x.T * x.dot(y.T).sum(-1) * x.dot(z.T).sum(-1)).sum(-1)
"""
