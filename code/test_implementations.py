
import functools

import torch
import torch.nn as nn

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp as lse
from jax.nn import softmax, log_softmax

import numpy as onp

from utils import renorm

import fast_attention as fat
import torch_fast_attention as tfat

from comparison import print_comp, report_mse, comp, print_comp_true

import plotly.graph_objects as go

import streamlit as st

num_features = 512
qk_dim = 64
T = 8

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

print("Gaussian Q K")
#true_attn, samples, gsamples, nsamples = comp(num_features, q, k, key)
print_comp(num_features, q, k, key)

sqrt_temp = 1.2
print(f"Gaussian Q K / {sqrt_temp}")
#true_attn, samples, gsamples, nsamples = comp(num_features, q / sqrt_temp, k / sqrt_temp, key)
print_comp(num_features, q / sqrt_temp, k / sqrt_temp, key)

print("observation: bad approximation in general because of BIAS with low entropy.")
print("conclusion: not a good idea to directly approximate with RFF, verifies experiments in RFF MT")
print("observation: approximation gets better when scaling embeddings by scaling towards 0")
print("possible explanation: need large number of samples for linear approximation of exponential?")
print("possible fix: keep values low with clamping or renorm?")

