
import torch
import torch.nn as nn

import jax
import jax.numpy as jnp

import numpy as onp

import fast_attention as fat
import torch_fast_attention as tfat

from comparison import comp, report_mse

num_features = 256
qk_dim = 8
T = 4

key = jax.random.PRNGKey(0)

key, key1, key2 = jax.random.split(key, 3)
q = jax.random.normal(key1, (T, qk_dim))
k = jax.random.normal(key2, (T, qk_dim))

q0, k0 = q.copy(), k.copy()

key, sample_key, norm_key  = jax.random.split(key, 3)
gaussian_sample = fat.random_projection(num_features, qk_dim, sample_key)
projection_matrix = fat.get_2d_array(gaussian_sample, norm_key)

# compare all attention implementations

## mean
vals = jnp.exp(q @ k.T)
true_attn = vals / vals.sum(-1, keepdims=True)

ra, _ = fat.rff_attn(q, k, projection_matrix)
ra0 = fat.rff_attn0(q, k, projection_matrix)

qt = torch.tensor(onp.asarray(q))
kt = torch.tensor(onp.asarray(k))
pt = torch.tensor(onp.asarray(projection_matrix)).transpose(-1, -2)

qf = tfat.kernel(qt[None], pt, is_query=True, eps=0)
kf = tfat.kernel(kt[None], pt, is_query=False, eps=0)
values = qf.bmm(kf.transpose(-1, -2))
attn = values / values.sum(-1, keepdim=True)

#import pdb; pdb.set_trace()

## var
num_samples = 128

samples = []
samples0 = []
gsamples = []
tsamples = []
nsamples = []
for i in range(num_samples):
    key, sample_key, norm_key  = jax.random.split(key, 3)
    gaussian_sample = fat.random_projection(num_features, qk_dim, sample_key)
    projection_matrix = fat.get_2d_array(gaussian_sample, norm_key)
    pt = torch.tensor(onp.asarray(projection_matrix)).transpose(-1, -2)

    ra, _ = fat.rff_attn(q, k, projection_matrix)
    ra0 = fat.rff_attn0(q, k, projection_matrix)
    samples.append(ra)
    samples0.append(ra0)

    gra, _ = fat.rff_attn(q, k, gaussian_sample)
    gsamples.append(gra)

    nprojection_matrix = fat.get_2d_array(gaussian_sample, norm_key, scaling=1)
    nra, _ = fat.rff_attn(q, k, nprojection_matrix)
    nsamples.append(nra)

    query_features = tfat.kernel(qt[None], pt, is_query=True, eps=0)
    key_features = tfat.kernel(kt[None], pt, is_query=False, eps=0)

    values = torch.bmm(query_features, key_features.transpose(-1, -2))
    attn = values / values.sum(-1, keepdim=True)
    tsamples.append(attn)

onp.set_printoptions(suppress=True,precision=4)

def report(sample):
    print(f"num_features {num_features}, emb dim {qk_dim}")
    print("variance")
    print(sample.var(0))
    print("mean")
    print(sample.mean(0))

print("jax")
report(jnp.stack(samples))
print("jax0")
report(jnp.stack(samples0))
print("torch")
report(torch.stack(tsamples))
print("true")
print(true_attn)

print("satisfied all versions are close")

# experiments with geometry of projection vs bias / variance

print("Geometry of proj")

true_attn, samples, gsamples, nsamples = comp(num_features, q, k, key)

print("true")
print(true_attn)
print("orth")
report_mse(samples, true_attn)
print("norm orth")
report_mse(nsamples, true_attn)
print("gaussian")
report_mse(gsamples, true_attn)

print("conclusion: bad approximation in general because of BIAS.")


