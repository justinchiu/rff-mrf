
import functools

import torch
import torch.nn as nn

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp as lse
from jax.nn import softmax, log_softmax

import numpy as onp

from utils import renorm, renorm_stopgrad

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

#num_iters = 1250
num_iters = 2500
#num_iters = 5000
num_iters = 10000
#num_iters = 20000
#alpha = 0.25
#alpha = 0.05
alpha = 1.

gamma = 0.001

def report_train(
    q, k, proj_fn, L_dL, num_features, key, train_fn, sample=True, title=None,
    post_renorm=False,
):
    vals = jnp.exp(q @ k.T)
    true_attn = vals / vals.sum(-1, keepdims=True)

    # sample embeddings close to 0 to start
    #key, key_q, key_k, key_pq, key_pk = jax.random.split(key, 5)
    key, key_q, key_k = jax.random.split(key, 3)
    q_init = jax.random.uniform(key_q, shape=q.shape, minval=-gamma, maxval=gamma)
    k_init = jax.random.uniform(key_k, shape=k.shape, minval=-gamma, maxval=gamma)

    if post_renorm:
        q_init = renorm(q_init)
        k_init = renorm(k_init)

    #scale = 1.
    #scale_q = jax.numpy.ones((num_features, 1))
    # seed 1
    key_pq = jax.random.PRNGKey(1111)
    # seed 2
    #key_pq = jax.random.PRNGKey(1234)
    proj_init_q = jax.device_put(proj_fn(key_pq))

    #scale_k = jax.numpy.ones((num_features, 1))
    #proj_init_k = proj_fn(key_pk)

    scale = 1.
    #scale = scale_q
    proj_init = proj_init_q

    key, key_train = jax.random.split(key)

    losses, grads, q_t, k_t, scale, proj = train_fn(
        #q_init.copy(), k_init.copy(),
        q_init, k_init,
        scale, proj_init if proj_init is not None else None,
        true_attn, L_dL, proj_fn,
        alpha, num_iters,
        key_train, sample,
        post_renorm=post_renorm,
    )
    #print(f"scale {scale}")
    #print(proj)
    #import pdb; pdb.set_trace()

    #"""
    fig = go.Figure(
        data = go.Scattergl(
            x = onp.arange(num_iters),
            y = losses,
            mode = "markers",
        ),
    )
    fig.update_layout(title = title)
    st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure(
        data = go.Scattergl(
            x = onp.arange(num_iters),
            y = grads,
            mode = "markers",
        ),
    )
    fig.update_layout(title = f"||GRAD||^2 {title}")
    st.plotly_chart(fig, use_container_width=True)
    #"""

    """
    if sample:
        key, key_comp = jax.random.split(key)
    else:
        key_comp = key_train
    print_comp_true(num_features, q_t, k_t, true_attn, key_comp, sample)
    """

    return losses[-32:].mean()#, mse 


key_train_init = jax.random.PRNGKey(12345)


def inner(num_features, qk_dim, S, T, temp_sqrt):
    qk_key = jax.random.PRNGKey(111)
    key1, key2 = jax.random.split(qk_key)
    q = jax.random.normal(key1, (S, qk_dim)) / temp_sqrt
    #q = jax.random.normal(key1, (T, qk_dim)) / temp_sqrt
    k = jax.random.normal(key2, (T, qk_dim)) / temp_sqrt

    log_probs = log_softmax(q @ k.T)
    print(f"Total entropy of true dist: {-(jnp.exp(log_probs) * log_probs).sum()}")
    print(f"Mean entropy of true dist: {-(jnp.exp(log_probs) * log_probs).sum(-1).mean()}")
    st.write(f"Total entropy of true dist: {-(jnp.exp(log_probs) * log_probs).sum()}")
    st.write(f"Mean entropy of true dist: {-(jnp.exp(log_probs) * log_probs).sum(-1).mean()}")

    #"""
    # fit exp kernel
    def loss(q, k, dummy_proj, attn):
        logits = q @ k.T
        probs = softmax(logits)
        return fat.kl(attn, probs).mean()
    L_dL = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))

    key, key1 = jax.random.split(key_train_init)
    print(f"Softmax fit")
    title = f"KL softmax fit (S: {S} T: {T} dim: {qk_dim} temp: {temp_sqrt})"
    kl_ = report_train(q, k, lambda x: None, L_dL, num_features, key1,
        train_fn = fat.train,
        sample=False, title=title,
    )
    print(f"kl {kl_}")
    #"""
    #
    """
    # fit exp kernel fwd renorm
    def loss(q, k, scale, dummy_proj, attn):
        logits = renorm(q) @ renorm(k).T
        probs = softmax(logits)
        return fat.kl(attn, probs).mean()
    L_dL = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3)))

    key, key1 = jax.random.split(key_train_init)
    print(f"Softmax fwd renorm fit")
    title = f"KL softmax fwd renorm fit (S: {S} T: {T} dim: {qk_dim} temp: {temp_sqrt})"
    kl_ = report_train(
        q, k,
        lambda x: None, L_dL, num_features, key1,
        train_fn = fat.train_proj,
        sample=False, title=title,
        post_renorm=False,
    )
    print(f"kl {kl_}")
    """
    
    """
    # fit exp kernel post renorm
    def loss(q, k, scale, dummy_proj, attn):
        logits = q @ k.T
        probs = softmax(logits)
        return fat.kl(attn, probs).mean()
    L_dL = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3)))

    key, key1 = jax.random.split(key_train_init)
    print(f"Softmax post renorm fit")
    title = f"KL softmax post renorm fit (S: {S} T: {T} dim: {qk_dim} temp: {temp_sqrt})"
    kl_ = report_train(
        q, k,
        lambda x: None, L_dL, num_features, key1,
        train_fn = fat.train_proj,
        sample=False, title=title,
        post_renorm=True,
    )
    print(f"kl {kl_}")
    """
    #
    """
    def loss(q, k, scale, dummy_proj, attn):
        logits = renorm(q) @ renorm(k).T
        probs = softmax(3. * logits)
        return fat.kl(attn, probs).mean()
    L_dL = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3)))

    key, key1 = jax.random.split(key_train_init)
    print(f"Softmax temp fwd renorm fit")
    title = f"KL softmax temp fwd renorm fit (S: {S} T: {T} dim: {qk_dim} temp: {temp_sqrt})"
    kl_ = report_train(
        q, k,
        lambda x: None, L_dL, num_features, key1,
        train_fn = fat.train_proj,
        sample=False, title=title,
        post_renorm=False,
    )
    print(f"kl {kl_}")
    """

    """
    def loss(q, k, scale, dummy_proj, attn):
        logits = q @ k.T
        probs = softmax(3. * logits)
        return fat.kl(attn, probs).mean()
    L_dL = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3)))

    key, key1 = jax.random.split(key_train_init)
    print(f"Softmax temp post renorm fit")
    title = f"KL softmax temp post renorm fit (S: {S} T: {T} dim: {qk_dim} temp: {temp_sqrt})"
    kl_ = report_train(
        q, k,
        lambda x: None, L_dL, num_features, key1,
        train_fn = fat.train_proj,
        sample=False, title=title,
        post_renorm=True,
    )
    print(f"kl {kl_}")
    """

    def proj_fn(shape, key):
        sample_key, norm_key  = jax.random.split(key)
        gaussian_sample = fat.random_projection(num_features, qk_dim, sample_key)
        print(gaussian_sample.shape)
        projection_matrix = fat.get_2d_array(gaussian_sample, norm_key, scaling=0)
        return projection_matrix
    proj_fn = functools.partial(proj_fn, (num_features, qk_dim))

    def proj_fn_gaus(shape, key):
        sample_key, norm_key  = jax.random.split(key)
        gaussian_sample = fat.random_projection(num_features, qk_dim, sample_key)
        return gaussian_sample
    proj_fn_gaus = functools.partial(proj_fn_gaus, (num_features, qk_dim))

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

    def proj_fn_reg_small(shape, key):
        sample_key, norm_key  = jax.random.split(key)
        gaussian_sample = fat.random_projection(num_features, qk_dim, sample_key)
        projection_matrix = fat.get_2d_array(gaussian_sample, norm_key, scaling=2)
        return projection_matrix
    proj_fn_reg_small = functools.partial(proj_fn_reg_small, (num_features, qk_dim))

    #for sample_key in [True, False]:
    for sample_key in [False]:
        #for proj_fn in [proj_fn, proj_fn_reg]:
        #for this_proj_fn in [proj_fn, proj_fn_anti]:
        for this_proj_fn in [proj_fn]:
            print(this_proj_fn)
            """
            def loss(q, k, scale, proj, attn_dist):
                ra, _ = fat.rff_attn(q, k, jax.lax.stop_gradient(proj))
                return fat.kl(attn_dist, ra).mean()
            L_dL = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3)))

            key, key1 = jax.random.split(key_train_init)
            print(f"Orth fit (Sample: {sample_key})")
            title = f"KL Orth fit (Sample: {sample_key} S: {S} T: {T} dim: {qk_dim} temp: {temp_sqrt} numfeat: {num_features})"
            kl_ = report_train(q, k, this_proj_fn, L_dL, num_features, key1,
                fat.train_proj, sample_key, title,
            )
            print(f"kl {kl_}")
            """

            #"""
            def loss(q, k, scale, proj, attn_dist):
                qp = renorm(q, axis=-1)
                kp = renorm(k, axis=-1)
                ra, _ = fat.rff_attn(qp, kp, jax.lax.stop_gradient(proj))
                return fat.kl(attn_dist, ra).mean()
            L_dL = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3)))

            key, key1 = jax.random.split(key_train_init)
            print(f"Orth projected L2 fit (Sample: {sample_key})")
            title = f"KL Orth projected L2 fit (Sample: {sample_key} S: {S} T: {T} dim: {qk_dim} temp: {temp_sqrt} numfeat: {num_features})"
            kl_ = report_train(q, k, proj_fn, L_dL, num_features, key1,
                fat.train_proj, sample_key, title,
            )
            print(f"kl {kl_}")
            #"""

            """
            def loss(q, k, scale, proj, attn_dist):
                qp = renorm_stopgrad(q, axis=-1)
                kp = renorm_stopgrad(k, axis=-1)
                ra, _ = fat.rff_attn(qp, kp, jax.lax.stop_gradient(proj))
                return fat.kl(attn_dist, ra).mean()
            L_dL = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3)))

            key, key1 = jax.random.split(key_train_init)
            print(f"Orth projected L2 detach fit (Sample: {sample_key})")
            title = f"KL Orth projected L2 detach fit (Sample: {sample_key} S: {S} T: {T} dim: {qk_dim} temp: {temp_sqrt} numfeat: {num_features})"
            kl_ = report_train(q, k, proj_fn, L_dL, num_features, key1,
                fat.train_proj, sample_key, title,
            )
            print(f"kl {kl_}")
            """

            """
            def loss(q, k, scale, proj, attn_dist):
                ra, _ = fat.rff_attn(q, k, jax.lax.stop_gradient(proj))
                return fat.kl(attn_dist, ra).mean()
            L_dL = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3)))

            key, key1 = jax.random.split(key_train_init)
            print(f"Orth post projected L2 fit (Sample: {sample_key})")
            title = f"KL orth post projected L2 fit (Sample: {sample_key} S: {S} T: {T} dim: {qk_dim} temp: {temp_sqrt} numfeat: {num_features})"
            kl_ = report_train(q, k, proj_fn, L_dL, num_features, key1,
                fat.train_proj, sample_key, title,
                post_renorm=True,
            )
            print(f"kl {kl_}")
            """

            #"""
            # learn scale
            def loss(q, k, scale, proj, attn_dist):
                qp = renorm(q, axis=-1)
                kp = renorm(k, axis=-1)
                ra, _ = fat.rff_attn(qp, kp, proj)
                return fat.kl(attn_dist, ra).mean()
            L_dL = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3)))

            key, key1 = jax.random.split(key_train_init)
            print(f"Orth projected L2 fit proj (Sample: {sample_key})")
            title = f"KL Orth Projected L2 fit proj (Sample: {sample_key} S: {S} T: {T} dim: {qk_dim} temp: {temp_sqrt} numfeat: {num_features})"
            kl_ = report_train(
                q, k,
                proj_fn,
                L_dL,
                num_features, key1,
                fat.train_proj, sample_key, title,
            )
            print(f"kl {kl_}")
            #"""

            """
            def loss(q, k, scale, proj, attn_dist):
                qp = renorm(q, axis=-1)
                kp = renorm(k, axis=-1)
                ra, _ = fat.rff_attn(qp, kp, jax.lax.stop_gradient(proj))
                return fat.kl(attn_dist, ra).mean()
            L_dL = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3)))

            key, key1 = jax.random.split(key_train_init)
            print(f"Orth Projected L2 fit scale projfnreg (Sample: {sample_key})")
            title = f"KL Orth Projected L2 fit scale projfnreg (Sample: {sample_key} S: {S} T: {T} dim: {qk_dim} temp: {temp_sqrt} numfeat: {num_features})"
            kl_ = report_train(
                q, k,
                #proj_fn,
                proj_fn_reg,
                L_dL,
                num_features, key1,
                fat.train_proj, sample_key, title,
            )
            print(f"kl {kl_}")
            """

            """
            def loss(q, k, scale, proj, attn_dist):
                qp = renorm(q, axis=-1)
                kp = renorm(k, axis=-1)
                ra, _ = fat.rff_attn(qp, kp, jax.lax.stop_gradient(scale) * proj)
                return fat.kl(attn_dist, ra).mean()
            L_dL = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3)))

            key, key1 = jax.random.split(key_train_init)
            print(f"Orth Projected L2 fit proj (Sample: {sample_key})")
            title = f"KL Orth Projected L2 fit proj (Sample: {sample_key} S: {S} T: {T} dim: {qk_dim} temp: {temp_sqrt} numfeat: {num_features})"
            kl_ = report_train(
                q, k,
                proj_fn,
                #proj_fn_norm,
                L_dL, num_features, key1,
                fat.train_proj, sample_key, title,
            )
            print(f"kl {kl_}")
            """

            """
            def loss(q, k, scale, proj, attn_dist):
                qp = q
                kp = k
                ra, _ = fat.relu_rff_attn0(qp, kp, jax.lax.stop_gradient(proj))
                return fat.kl(attn_dist, ra).mean()
            L_dL = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3)))
            #L_dL = jax.value_and_grad(loss, argnums=(0, 1, 2, 3)

            key, key1 = jax.random.split(key_train_init)
            #print(f"Relu0 Projected L2 fit (Sample: {sample_key})")
            #title = f"KL Relu0 Projected L2 fit (Sample: {sample_key} S: {S} T: {T} dim: {qk_dim} temp: {temp_sqrt} numfeat: {num_features})"
            print(f"Relu0 fit (Sample: {sample_key})")
            title = f"KL Relu0 fit (Sample: {sample_key} S: {S} T: {T} dim: {qk_dim} temp: {temp_sqrt} numfeat: {num_features})"
            kl_ = report_train(
                q, k,
                #proj_fn,
                #proj_fn_gaus,
                proj_fn_reg,
                L_dL, num_features, key1,
                fat.train_proj, sample_key, title,
            )
            print(f"kl {kl_}")
            """
            #"""
            def loss(q, k, scale, proj, attn_dist):
                qp = q
                kp = k
                ra, _ = fat.exp_rff_attn(qp, kp, jax.lax.stop_gradient(proj))
                return fat.kl(attn_dist, ra).mean()
            L_dL = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3)))
            #L_dL = jax.value_and_grad(loss, argnums=(0, 1, 2, 3)

            key, key1 = jax.random.split(key_train_init)
            print(f"Exp fit (Sample: {sample_key})")
            title = f"KL Exp fit (Sample: {sample_key} S: {S} T: {T} dim: {qk_dim} temp: {temp_sqrt} numfeat: {num_features})"
            kl_ = report_train(
                q, k,
                #proj_fn,
                #proj_fn_gaus,
                proj_fn_reg,
                L_dL, num_features, key1,
                fat.train_proj, sample_key, title,
            )
            print(f"kl {kl_}")
            #"""
            #
            #"""
            def loss(q, k, scale, proj, attn_dist):
                qp = q
                kp = k
                ra, _ = fat.relu_rff_attn(qp, kp, jax.lax.stop_gradient(proj))
                return fat.kl(attn_dist, ra).mean()
            L_dL = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3)))
            #L_dL = jax.value_and_grad(loss, argnums=(0, 1, 2, 3)

            key, key1 = jax.random.split(key_train_init)
            #print(f"Relu Projected L2 fit (Sample: {sample_key})")
            #title = f"KL Relu Projected L2 fit (Sample: {sample_key} S: {S} T: {T} dim: {qk_dim} temp: {temp_sqrt} numfeat: {num_features})"
            print(f"Relu fit (Sample: {sample_key})")
            title = f"KL Relu fit (Sample: {sample_key} S: {S} T: {T} dim: {qk_dim} temp: {temp_sqrt} numfeat: {num_features})"
            kl_ = report_train(
                q, k,
                #proj_fn,
                #proj_fn_gaus,
                proj_fn_reg,
                L_dL, num_features, key1,
                fat.train_proj, sample_key, title,
            )
            print(f"kl {kl_}")
            #"""
            #"""
            def loss(q, k, scale, proj, attn_dist):
                qp = q
                kp = k
                ra, _ = fat.relu_rff_attn(qp, kp, jax.lax.stop_gradient(proj))
                return fat.kl(attn_dist, ra).mean()
            L_dL = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3)))
            #L_dL = jax.value_and_grad(loss, argnums=(0, 1, 2, 3)

            key, key1 = jax.random.split(key_train_init)
            #print(f"Relu Projected L2 fit (Sample: {sample_key})")
            #title = f"KL Relu Projected L2 fit (Sample: {sample_key} S: {S} T: {T} dim: {qk_dim} temp: {temp_sqrt} numfeat: {num_features})"
            print(f"Relu fit small (Sample: {sample_key})")
            title = f"KL Relu fit small (Sample: {sample_key} S: {S} T: {T} dim: {qk_dim} temp: {temp_sqrt} numfeat: {num_features})"
            kl_ = report_train(
                q, k,
                #proj_fn,
                #proj_fn_gaus,
                proj_fn_reg_small,
                L_dL, num_features, key1,
                fat.train_proj, sample_key, title,
            )
            print(f"kl {kl_}")
            #"""
            #
            #
            #"""
            def loss(q, k, scale, proj, attn_dist):
                qp = q
                kp = k
                ra, _ = fat.relu_rff_attn(qp, kp, proj)
                return fat.kl(attn_dist, ra).mean()
            L_dL = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3)))
            #L_dL = jax.value_and_grad(loss, argnums=(0, 1, 2, 3)

            key, key1 = jax.random.split(key_train_init)
            #print(f"Relu Projected L2 fit (Sample: {sample_key})")
            #title = f"KL Relu Projected L2 fit (Sample: {sample_key} S: {S} T: {T} dim: {qk_dim} temp: {temp_sqrt} numfeat: {num_features})"
            print(f"Relu fit small proj (Sample: {sample_key})")
            title = f"KL Relu fit proj (Sample: {sample_key} S: {S} T: {T} dim: {qk_dim} temp: {temp_sqrt} numfeat: {num_features})"
            kl_ = report_train(
                q, k,
                #proj_fn,
                #proj_fn_gaus,
                proj_fn_reg_small,
                L_dL, num_features, key1,
                fat.train_proj, sample_key, title,
            )
            print(f"kl {kl_}")
            #"""
            #
            #"""
            def loss(q, k, scale, proj, attn_dist):
                qp = q
                kp = k
                ra, _ = fat.relu_rff_attn(qp, kp, scale * proj)
                return fat.kl(attn_dist, ra).mean()
            L_dL = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3)))
            #L_dL = jax.value_and_grad(loss, argnums=(0, 1, 2, 3)

            key, key1 = jax.random.split(key_train_init)
            #print(f"Relu Projected L2 fit (Sample: {sample_key})")
            #title = f"KL Relu Projected L2 fit (Sample: {sample_key} S: {S} T: {T} dim: {qk_dim} temp: {temp_sqrt} numfeat: {num_features})"
            print(f"Relu fit small proj (Sample: {sample_key})")
            title = f"KL Relu fit small proj scale (Sample: {sample_key} S: {S} T: {T} dim: {qk_dim} temp: {temp_sqrt} numfeat: {num_features})"
            kl_ = report_train(
                q, k,
                #proj_fn,
                #proj_fn_gaus,
                proj_fn_reg_small,
                L_dL, num_features, key1,
                fat.train_proj, sample_key, title,
            )
            print(f"kl {kl_}")
            #"""

            """
            # bad
            def loss(q, k, scale, proj, attn_dist):
                qp = renorm(q)
                kp = renorm(k)
                ra, _ = fat.relu_rff_attn(qp, kp, proj)
                return fat.kl(attn_dist, ra).mean()
            L_dL = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3)))
            #L_dL = jax.value_and_grad(loss, argnums=(0, 1, 2, 3)

            key, key1 = jax.random.split(key_train_init)
            #print(f"Relu Projected L2 fit (Sample: {sample_key})")
            #title = f"KL Relu Projected L2 fit (Sample: {sample_key} S: {S} T: {T} dim: {qk_dim} temp: {temp_sqrt} numfeat: {num_features})"
            print(f"Relu projected fit proj (Sample: {sample_key})")
            title = f"KL Relu projected fit proj (Sample: {sample_key} S: {S} T: {T} dim: {qk_dim} temp: {temp_sqrt} numfeat: {num_features})"
            kl_ = report_train(
                q, k,
                #proj_fn,
                #proj_fn_gaus,
                proj_fn_reg,
                L_dL, num_features, key1,
                fat.train_proj, sample_key, title,
            )
            print(f"kl {kl_}")
            """

            """
            def loss(q, k, scale, proj, attn_dist):
                #qp = renorm(q)
                #kp = renorm(k)
                qp = q
                kp = k
                ra, _ = fat.sincos_rff_attn0(qp, kp, 4. * jax.lax.stop_gradient(proj))
                return fat.kl(attn_dist, ra).mean()
            L_dL = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3)))
            #L_dL = jax.value_and_grad(loss, argnums=(0, 1, 2, 3))

            key, key1 = jax.random.split(key_train_init)
            print(f"Sincos fit (Sample: {sample_key})")
            title = f"KL Sincosfit (Sample: {sample_key} S: {S} T: {T} dim: {qk_dim} temp: {temp_sqrt} numfeat: {num_features})"
            kl_ = report_train(
                q, k,
                proj_fn,
                #proj_fn_reg,
                L_dL,
                #num_features // 2, # since projection doubles in size?
                num_features,
                key1,
                fat.train_proj, sample_key, title,
                post_renorm = True,
            )
            print(f"kl {kl_}")
            """




#for num_features in [256, 512]:
for num_features in [256]:
    #for temp in [1, 1.25, 1.5]:
    for temp in [1]:
        #for qk_dim in [64, 128]:
        #for qk_dim in [64]:
        for qk_dim in [128]:
            #for T in [8, 32, 128, 256]:
            #for T in [8]:
            #for S, T in [(64, 1024), (256, 1024)]:
            for S, T in [(256, 1024)]:
                print(f"num_features {num_features} qk_dim {qk_dim} S {S} T {T} temp_sqrt {temp}")
                inner(num_features, qk_dim, S, T, temp)
