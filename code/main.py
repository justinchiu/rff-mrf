
import functools

import torch
import torch.nn as nn

import jax
import jax.numpy as jnp

import numpy as onp

from utils import renorm

import fast_attention as fat
import torch_fast_attention as tfat

from comparison import print_comp, report_mse, comp, print_comp_true

import plotly.graph_objects as go

import streamlit as st

num_features = 256
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

print("Learn through approximation directly")
print("try to fit low entropy distributions")

num_iters = 2000
alpha = 0.25

def report_train(q, k, proj_fn, L_dL, key, sample=True):
    vals = jnp.exp(q @ k.T)
    true_attn = vals / vals.sum(-1, keepdims=True)

    key, key_train = jax.random.split(key)
    losses, q_t, k_t = fat.train(
        q.copy(), k.copy(),
        true_attn, L_dL, proj_fn,
        alpha, num_iters,
        key_train, sample,
    )
    fig = go.Figure(data=go.Scatter(
        x = onp.arange(num_iters),
        y = losses,
        mode = "markers",
    ))
    st.plotly_chart(fig, use_container_width=True)
    if sample:
        key, key_comp = jax.random.split(key)
    else:
        key_comp = key_train
    print_comp_true(num_features, q_t, k_t, true_attn, key_comp, sample)


key_train_init = key


def inner(num_features, qk_dim, T):
    qk_key = jax.random.PRNGKey(111)
    key1, key2 = jax.random.split(qk_key)
    q = jax.random.normal(key1, (T, qk_dim))
    k = jax.random.normal(key2, (T, qk_dim))

    def proj_fn(shape, key):
        sample_key, norm_key  = jax.random.split(key)
        gaussian_sample = fat.random_projection(num_features, qk_dim, sample_key)
        projection_matrix = fat.get_2d_array(gaussian_sample, norm_key, scaling=0)
        return projection_matrix
    proj_fn = functools.partial(proj_fn, (num_features, qk_dim))

    def proj_fn_reg(shape, key):
        sample_key, norm_key  = jax.random.split(key)
        gaussian_sample = fat.random_projection(num_features, qk_dim, sample_key)
        projection_matrix = fat.get_2d_array(gaussian_sample, norm_key, scaling=1)
        return projection_matrix
    proj_fn_reg = functools.partial(proj_fn_reg, (num_features, qk_dim))

    for sample_key in [True, False]:
        #for proj_fn in [proj_fn, proj_fn_reg]:
        for proj_fn in [proj_fn]:
            def loss(q, k, attn_dist, proj):
                # numerically unstable?
                ra, _ = fat.rff_attn(q, k, proj, eps=1e-6)
                return fat.kl(attn_dist, ra).mean()
            L_dL = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))

            key, key1 = jax.random.split(key_train_init)
            print(f"Normal fit (Sample: {sample_key})")
            report_train(q, k, proj_fn, L_dL, key1, sample_key)

            def loss(q, k, attn_dist, proj):
                qp = renorm(q, 2, axis=-1)
                kp = renorm(k, 2, axis=-1)
                ra, _ = fat.rff_attn(qp, kp, proj)
                return fat.kl(attn_dist, ra).mean()
            L_dL = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))

            key, key1 = jax.random.split(key_train_init)
            print(f"Projected L2 fit (Sample: {sample_key})")
            report_train(q, k, proj_fn, L_dL, key1, sample_key)

            def loss(q, k, attn_dist, proj):
                qp = jax.lax.clamp(-2., q, 2.)
                kp = jax.lax.clamp(-2., k, 2.)
                ra, _ = fat.rff_attn(qp, kp, proj)
                return fat.kl(attn_dist, ra).mean()
            L_dL = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))

            key, key1 = jax.random.split(key_train_init)
            print(f"Projected Linf fit (Sample: {sample_key})")
            report_train(q, k, proj_fn, L_dL, key1, sample_key)


num_features = 256
qk_dim = 64
T = 8

for qk_dim in [64, 128]:
    for T in [8, 32, 128]:
        print(f"num_features {num_features} qk_dim {qk_dim} T {T}")
        inner(num_features, qk_dim, T)
