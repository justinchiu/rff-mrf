import jax
import jax.numpy as jnp

import numpy as onp

import fast_attention as fat


def comp(num_features, q, k, key, num_samples=128):
    T, qk_dim = q.shape

    q0, k0 = q.copy(), k.copy()

    key, sample_key, norm_key  = jax.random.split(key, 3)
    gaussian_sample = fat.random_projection(num_features, qk_dim, sample_key)
    projection_matrix = fat.get_2d_array(gaussian_sample, norm_key)

    # compare all attention implementations
    vals = jnp.exp(q @ k.T)
    true_attn = vals / vals.sum(-1, keepdims=True)

    samples = []
    gsamples = []
    nsamples = []
    for i in range(num_samples):
        key, sample_key, norm_key  = jax.random.split(key, 3)
        gaussian_sample = fat.random_projection(num_features, qk_dim, sample_key)
        projection_matrix = fat.get_2d_array(gaussian_sample, norm_key)

        ra, _ = fat.rff_attn(q, k, projection_matrix)
        samples.append(ra)

        gra, _ = fat.rff_attn(q, k, gaussian_sample)
        gsamples.append(gra)

        nprojection_matrix = fat.get_2d_array(gaussian_sample, norm_key, scaling=1)
        nra, _ = fat.rff_attn(q, k, nprojection_matrix)
        nsamples.append(nra)

    return (true_attn,) + tuple(jnp.stack(x) for x in [samples, gsamples, nsamples])


def report_mse(sample, true):
    print("mean")
    print(sample.mean(0))
    print("variance")
    var = sample.var(0)
    print(var)
    print("bias ^ 2")
    bias = sample.mean(0) - true
    print(bias ** 2)
    print((bias ** 2 + var).sum())

def print_comp(num_features, q, k, key):
    true_attn, samples, gsamples, nsamples = comp(num_features, q, k, key)

    print("true")
    print(true_attn)
    print("orth")
    report_mse(samples, true_attn)
    print("norm orth")
    report_mse(nsamples, true_attn)
    print("gaussian")
    report_mse(gsamples, true_attn)

def print_comp_true(num_features, q, k, true_attn, key):
    _, samples, gsamples, nsamples = comp(num_features, q, k, key)

    print("true")
    print(true_attn)
    print("orth")
    report_mse(samples, true_attn)
    print("norm orth")
    report_mse(nsamples, true_attn)
    print("gaussian")
    report_mse(gsamples, true_attn)
