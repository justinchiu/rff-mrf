
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

onp.set_printoptions(suppress=True,precision=4)

def report(sample):
    print(f"num_features {num_features}, emb dim {qk_dim}")
    print("variance")
    print(sample.var(0))
    print("mean")
    print(sample.mean(0))

print("Learn through approximation directly")
print("try to fit low entropy distributions")

num_iters = 2500
alpha = 0.25

gamma = 0.001

def report_train(q, k, proj_fn, L_dL, num_features, key, sample=True, title=None):
    vals = jnp.exp(q @ k.T)
    true_attn = vals / vals.sum(-1, keepdims=True)

    # sample embeddings close to 0 to start
    key, key_q, key_k = jax.random.split(key, 3)
    q = jax.random.uniform(key_q, shape=q.shape, minval=-gamma, maxval=gamma)
    k = jax.random.uniform(key_k, shape=k.shape, minval=-gamma, maxval=gamma)

    key, key_train = jax.random.split(key)

    losses, q_t, k_t = fat.train(
        q.copy(), k.copy(),
        true_attn, L_dL, proj_fn,
        alpha, num_iters,
        key_train, sample,
    )

    fig = go.Figure(
        data = go.Scatter(
            x = onp.arange(num_iters),
            y = losses,
            mode = "markers",
        ),
    )
    fig.update_layout(title = title)
    st.plotly_chart(fig, use_container_width=True)

    if sample:
        key, key_comp = jax.random.split(key)
    else:
        key_comp = key_train
    print_comp_true(num_features, q_t, k_t, true_attn, key_comp, sample)

    return losses[-32:].mean()#, mse 


key_train_init = jax.random.PRNGKey(12345)


def inner(num_features, qk_dim, T, temp_sqrt):
    qk_key = jax.random.PRNGKey(111)
    key1, key2 = jax.random.split(qk_key)
    q = jax.random.normal(key1, (8, qk_dim)) / temp_sqrt
    #q = jax.random.normal(key1, (T, qk_dim)) / temp_sqrt
    k = jax.random.normal(key2, (T, qk_dim)) / temp_sqrt

    log_probs = log_softmax(q @ k.T)
    print(f"Total entropy of true dist: {-(jnp.exp(log_probs) * log_probs).sum()}")
    print(f"Mean entropy of true dist: {-(jnp.exp(log_probs) * log_probs).sum(-1).mean()}")
    st.write(f"Total entropy of true dist: {-(jnp.exp(log_probs) * log_probs).sum()}")
    st.write(f"Mean entropy of true dist: {-(jnp.exp(log_probs) * log_probs).sum(-1).mean()}")

    # fit exp kernel
    def loss(q, k, attn, dummy):
        logits = q @ k.T
        probs = softmax(logits)
        return fat.kl(attn, probs).mean()
    L_dL = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))

    key, key1 = jax.random.split(key_train_init)
    print(f"Softmax fit")
    title = f"KL softmax fit (T: {T} dim: {qk_dim} temp: {temp_sqrt})"
    kl_ = report_train(q, k, lambda x: None, L_dL, num_features, key1, False, title)
    print(f"kl {kl_}")

    def proj_fn(shape, key):
        sample_key, norm_key  = jax.random.split(key)
        gaussian_sample = fat.random_projection(num_features, qk_dim, sample_key)
        projection_matrix = fat.get_2d_array(gaussian_sample, norm_key, scaling=0)
        return projection_matrix
    proj_fn = functools.partial(proj_fn, (num_features, qk_dim))

    def proj_fn_anti(shape, key):
        sample_key, norm_key  = jax.random.split(key)
        gaussian_sample = fat.random_projection(num_features // 2, qk_dim, sample_key)
        projection_matrix = fat.get_2d_array(gaussian_sample, norm_key, scaling=0)
        return jnp.concatenate([projection_matrix, -projection_matrix], axis=0)
    proj_fn_anti = functools.partial(proj_fn_anti, (num_features, qk_dim))

    def proj_fn_reg(shape, key):
        sample_key, norm_key  = jax.random.split(key)
        gaussian_sample = fat.random_projection(num_features, qk_dim, sample_key)
        projection_matrix = fat.get_2d_array(gaussian_sample, norm_key, scaling=1)
        return projection_matrix
    proj_fn_reg = functools.partial(proj_fn_reg, (num_features, qk_dim))

    #for sample_key in [True, False]:
    for sample_key in [False]:
        #for proj_fn in [proj_fn, proj_fn_reg]:
        for proj_fn in [proj_fn, proj_fn_anti]:
            print(proj_fn)
            #"""
            def loss(q, k, attn_dist, proj):
                # numerically unstable?
                ra, _ = fat.rff_attn(q, k, proj, eps=1e-6)
                return fat.kl(attn_dist, ra).mean()
            L_dL = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))

            key, key1 = jax.random.split(key_train_init)
            print(f"Normal fit (Sample: {sample_key})")
            title = f"KL Normal fit (Sample: {sample_key} T: {T} dim: {qk_dim} temp: {temp_sqrt} numfeat: {num_features})"
            kl_ = report_train(q, k, proj_fn, L_dL, num_features, key1, sample_key, title)
            print(f"kl {kl_}")
            #"""

            #"""
            def loss(q, k, attn_dist, proj):
                qp = renorm(q, 2, axis=-1)
                kp = renorm(k, 2, axis=-1)
                ra, _ = fat.rff_attn(qp, kp, proj)
                return fat.kl(attn_dist, ra).mean()
            L_dL = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))

            key, key1 = jax.random.split(key_train_init)
            print(f"Projected L2 fit (Sample: {sample_key})")
            title = f"KL Projected L2 fit (Sample: {sample_key} T: {T} dim: {qk_dim} temp: {temp_sqrt} numfeat: {num_features})"
            kl_ = report_train(q, k, proj_fn, L_dL, num_features, key1, sample_key, title)
            print(f"kl {kl_}")


num_features = 256
num_features = 512
for num_features in [256, 512]:
    #for temp in [1, 1.25, 1.5]:
    for temp in [1]:
        for qk_dim in [64, 128]:
        #for qk_dim in [64]:
        #for qk_dim in [128]:
            #for T in [8, 32, 128, 256]:
            #for T in [8]:
            for T in [4096]:
                print(f"num_features {num_features} qk_dim {qk_dim} T {T} temp_sqrt {temp}")
                inner(num_features, qk_dim, T, temp)
