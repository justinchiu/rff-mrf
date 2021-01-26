# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=invalid-name, missing-function-docstring, line-too-long

import abc
from collections.abc import Iterable  # pylint: disable=g-importing-member
import functools
import logging
import math

import jax
from jax import lax
from jax import random
import jax.numpy as jnp

from jax.scipy.special import logsumexp as lse

from jax.numpy.linalg import norm

import numpy as onp

from utils import renorm


def nonnegative_softmax_kernel_feature_creator(
    data,
    projection_matrix,
    is_query,
    normalize_data=True,
    eps=0.0001,
):
    """Constructs nonnegative kernel features for fast softmax attention.


    Args:
    data: input for which features are computes
    projection_matrix: random matrix used to compute features
    batch_dims_t: tuple of batch dimensions
    is_query: predicate indicating whether input data corresponds to queries or
      keys
    normalize_data: predicate indicating whether data should be normalized,
    eps: numerical stabilizer.

    Returns:
    Random features for fast softmax attention.
    """
    # We have e^{qk^T/sqrt{d}} = e^{q_norm k_norm^T} (transformer), where
    # w_norm = w * data_normalizer for w in {q,k}.
    data_normalizer = 1.0 / (jnp.sqrt(jnp.sqrt(data.shape[-1]))) if normalize_data else 1.0

    # ratio is for computing the empirical mean / avg
    # its a constant that gets cancelled out in softmax though?
    #ratio = 1.0 / jnp.sqrt(projection_matrix.shape[0])
    ratio = 1.0

    # matmul / linear projection
    data_dash = jnp.einsum("...bd,...fd->...bf", data_normalizer * data, projection_matrix)

    diag_data = jnp.square(data)
    diag_data = jnp.sum(diag_data, axis=data.ndim - 1)
    diag_data = (diag_data / 2.0) * data_normalizer * data_normalizer
    diag_data = jnp.expand_dims(diag_data, axis=data.ndim - 1)

    return data_dash - diag_data + eps - math.log(ratio)

def nonnegative_softmax_kernel_feature_creator0(
    data,
    projection_matrix,
    batch_dims_t,
    precision,
    is_query,
    normalize_data=True,
    eps=0.0001,
):
  """Constructs nonnegative kernel features for fast softmax attention.

  Args:
    data: input for which features are computes
    projection_matrix: random matrix used to compute features
    batch_dims_t: tuple of batch dimensions
    precision: precision parameter
    is_query: predicate indicating whether input data corresponds to queries or
      keys
    normalize_data: predicate indicating whether data should be normalized,
    eps: numerical stabilizer.

  Returns:
    Random features for fast softmax attention.
  """
  if normalize_data:
    # We have e^{qk^T/sqrt{d}} = e^{q_norm k_norm^T}, where
    # w_norm = w * data_normalizer for w in {q,k}.
    data_normalizer = 1.0 / (jnp.sqrt(jnp.sqrt(data.shape[-1])))
  else:
    data_normalizer = 1.0
  #ratio = 1.0 / jnp.sqrt(projection_matrix.shape[0])
  ratio = 1.0
  data_mod_shape = data.shape[0:len(batch_dims_t)] + projection_matrix.shape
  data_thick_random_matrix = jnp.zeros(data_mod_shape) + projection_matrix

  #"""
  data_dash = lax.dot_general(
      data_normalizer * data,
      data_thick_random_matrix,
      (((data.ndim - 1,), (data_thick_random_matrix.ndim - 1,)),
       (batch_dims_t, batch_dims_t)),
      precision=precision)
  #"""
  #data_dash = jnp.einsum("...bd,...fd->...bf", data_normalizer * data, projection_matrix)

  diag_data = jnp.square(data)
  diag_data = jnp.sum(diag_data, axis=data.ndim - 1)
  diag_data = (diag_data / 2.0) * data_normalizer * data_normalizer
  diag_data = jnp.expand_dims(diag_data, axis=data.ndim - 1)

  if is_query:
    last_dims_t = (len(data_dash.shape) - 1,)
    data_dash = ratio * (
        jnp.exp(data_dash - diag_data -
                jnp.max(data_dash, axis=last_dims_t, keepdims=True)) + eps)
  else:
    data_dash = ratio * (
        jnp.exp(data_dash - diag_data - jnp.max(data_dash)) + eps)

  return data_dash

def relu_nonnegative_softmax_kernel_feature_creator0(
    data,
    projection_matrix,
    batch_dims_t,
    precision,
    is_query,
    normalize_data=False,
    eps=0.0001,
):
  """Constructs nonnegative kernel features for fast softmax attention.

  Args:
    data: input for which features are computes
    projection_matrix: random matrix used to compute features
    batch_dims_t: tuple of batch dimensions
    precision: precision parameter
    is_query: predicate indicating whether input data corresponds to queries or
      keys
    normalize_data: predicate indicating whether data should be normalized,
    eps: numerical stabilizer.

  Returns:
    Random features for fast softmax attention.
  """
  if normalize_data:
    # We have e^{qk^T/sqrt{d}} = e^{q_norm k_norm^T}, where
    # w_norm = w * data_normalizer for w in {q,k}.
    data_normalizer = 1.0 / (jnp.sqrt(jnp.sqrt(data.shape[-1])))
  else:
    data_normalizer = 1.0
  #ratio = 1.0 / jnp.sqrt(projection_matrix.shape[0])
  ratio = 1.0
  data_mod_shape = data.shape[0:len(batch_dims_t)] + projection_matrix.shape
  data_thick_random_matrix = jnp.zeros(data_mod_shape) + projection_matrix

  #"""
  data_dash = lax.dot_general(
      data_normalizer * data,
      data_thick_random_matrix,
      (((data.ndim - 1,), (data_thick_random_matrix.ndim - 1,)),
       (batch_dims_t, batch_dims_t)),
      precision=precision)
  #"""
  #data_dash = jnp.einsum("...bd,...fd->...bf", data_normalizer * data, projection_matrix)

  diag_data = jnp.square(data)
  diag_data = jnp.sum(diag_data, axis=data.ndim - 1)
  diag_data = (diag_data / 2.0) * data_normalizer * data_normalizer
  diag_data = jnp.expand_dims(diag_data, axis=data.ndim - 1)

  if is_query:
    last_dims_t = (len(data_dash.shape) - 1,)
    data_dash = ratio * (
        #jnp.exp(
            #data_dash - diag_data -
            #jnp.max(data_dash, axis=last_dims_t, keepdims=True)) + eps)
        jax.nn.relu(data_dash)) + eps
  else:
    data_dash = ratio * (
        #jnp.exp(
            #data_dash - diag_data - jnp.max(data_dash)) + eps)
        jax.nn.relu(data_dash)) + eps

  return data_dash

def relu_nonnegative_softmax_kernel_feature_creator(
    data,
    projection_matrix,
    batch_dims_t,
    precision,
    is_query,
    normalize_data=False,
    eps=0.0001,
):
  """Constructs nonnegative kernel features for fast softmax attention.

  Args:
    data: input for which features are computes
    projection_matrix: random matrix used to compute features
    batch_dims_t: tuple of batch dimensions
    precision: precision parameter
    is_query: predicate indicating whether input data corresponds to queries or
      keys
    normalize_data: predicate indicating whether data should be normalized,
    eps: numerical stabilizer.

  Returns:
    Random features for fast softmax attention.
  """
  if normalize_data:
    # We have e^{qk^T/sqrt{d}} = e^{q_norm k_norm^T}, where
    # w_norm = w * data_normalizer for w in {q,k}.
    data_normalizer = 1.0 / (jnp.sqrt(jnp.sqrt(data.shape[-1])))
  else:
    data_normalizer = 1.0
  #ratio = 1.0 / jnp.sqrt(projection_matrix.shape[0])
  ratio = 1.0
  data_mod_shape = data.shape[0:len(batch_dims_t)] + projection_matrix.shape
  data_thick_random_matrix = jnp.zeros(data_mod_shape) + projection_matrix

  #"""
  data_dash = lax.dot_general(
      data_normalizer * data,
      data_thick_random_matrix,
      (((data.ndim - 1,), (data_thick_random_matrix.ndim - 1,)),
       (batch_dims_t, batch_dims_t)),
      precision=precision)
  #"""
  #data_dash = jnp.einsum("...bd,...fd->...bf", data_normalizer * data, projection_matrix)
  #
  # instead of adding, try smoothing this way
  return jnp.log(jnp.maximum(data_dash, eps))

def exp_nonnegative_softmax_kernel_feature_creator(
    data,
    projection_matrix,
    batch_dims_t,
    precision,
    is_query,
    normalize_data=False,
    eps=0.0001,
):
  """Constructs nonnegative kernel features for fast softmax attention.

  Args:
    data: input for which features are computes
    projection_matrix: random matrix used to compute features
    batch_dims_t: tuple of batch dimensions
    precision: precision parameter
    is_query: predicate indicating whether input data corresponds to queries or
      keys
    normalize_data: predicate indicating whether data should be normalized,
    eps: numerical stabilizer.

  Returns:
    Random features for fast softmax attention.
  """
  if normalize_data:
    # We have e^{qk^T/sqrt{d}} = e^{q_norm k_norm^T}, where
    # w_norm = w * data_normalizer for w in {q,k}.
    data_normalizer = 1.0 / (jnp.sqrt(jnp.sqrt(data.shape[-1])))
  else:
    data_normalizer = 1.0
  #ratio = 1.0 / jnp.sqrt(projection_matrix.shape[0])
  ratio = 1.0
  data_mod_shape = data.shape[0:len(batch_dims_t)] + projection_matrix.shape
  data_thick_random_matrix = jnp.zeros(data_mod_shape) + projection_matrix

  #"""
  data_dash = lax.dot_general(
      data_normalizer * data,
      data_thick_random_matrix,
      (((data.ndim - 1,), (data_thick_random_matrix.ndim - 1,)),
       (batch_dims_t, batch_dims_t)),
      precision=precision)
  #"""
  #data_dash = jnp.einsum("...bd,...fd->...bf", data_normalizer * data, projection_matrix)
  return data_dash

def sincos_nonnegative_softmax_kernel_feature_creator0(
    data,
    projection_matrix,
    batch_dims_t,
    precision,
    is_query,
    normalize_data=False,
    eps=0.0001,
):
    """Constructs nonnegative kernel features for fast softmax attention.

    Args:
    data: input for which features are computes
    projection_matrix: random matrix used to compute features
    batch_dims_t: tuple of batch dimensions
    precision: precision parameter
    is_query: predicate indicating whether input data corresponds to queries or
      keys
    normalize_data: predicate indicating whether data should be normalized,
    eps: numerical stabilizer.

    Returns:
    Random features for fast softmax attention.
    """
    if normalize_data:
        # We have e^{qk^T/sqrt{d}} = e^{q_norm k_norm^T}, where
        # w_norm = w * data_normalizer for w in {q,k}.
        data_normalizer = 1.0 / (jnp.sqrt(jnp.sqrt(data.shape[-1])))
    else:
        data_normalizer = 1.0
    #ratio = 1.0 / jnp.sqrt(projection_matrix.shape[0])
    ratio = 1.0
    data_mod_shape = data.shape[0:len(batch_dims_t)] + projection_matrix.shape
    data_thick_random_matrix = jnp.zeros(data_mod_shape) + projection_matrix

    #"""
    data_dash = lax.dot_general(
      data_normalizer * data,
      data_thick_random_matrix,
      (((data.ndim - 1,), (data_thick_random_matrix.ndim - 1,)),
       (batch_dims_t, batch_dims_t)),
      precision=precision)
    #"""
    #data_dash = jnp.einsum("...bd,...fd->...bf", data_normalizer * data, projection_matrix)
    data_dash_cos = ratio * jnp.cos(data_dash)
    data_dash_sin = ratio * jnp.sin(data_dash)
    data_dash = jnp.concatenate((data_dash_cos, data_dash_sin), axis=-1)

    # Constructing D_data and data^{'}
    diag_data = jnp.square(data)
    diag_data = jnp.sum(diag_data, axis=data.ndim - 1)
    diag_data = (diag_data / 2.0) * data_normalizer * data_normalizer
    diag_data = jnp.expand_dims(diag_data, axis=data.ndim - 1)
    # Additional renormalization for numerical stability
    # which one?
    #data_renormalizer = jnp.max(diag_data, -1, keepdims=True)
    data_renormalizer = jnp.max(diag_data)
    diag_data -= data_renormalizer
    diag_data = jnp.exp(diag_data)
    data_prime = data_dash * diag_data

    return data_prime + eps


def get_2d_array(unstructured_blocks, key, scaling=0):
    nb_rows, nb_columns = unstructured_blocks.shape
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []
    rng = key
    for _ in range(nb_full_blocks):
      rng, rng_input = jax.random.split(rng)
      unstructured_block = random.normal(rng_input,
                                         (nb_columns, nb_columns))
      q, _ = jnp.linalg.qr(unstructured_block)
      q = jnp.transpose(q)
      block_list.append(q)
    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
      rng, rng_input = jax.random.split(rng)
      unstructured_block = random.normal(rng_input,
                                         (nb_columns, nb_columns))
      q, _ = jnp.linalg.qr(unstructured_block)
      q = jnp.transpose(q)
      block_list.append(q[0:remaining_rows])
    final_matrix = jnp.vstack(block_list)

    """
    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        raise ValueError("Assert nb_rows % nb_columns == 0 for simplicity")
        # if want to change this take a look at the garbage in fast_attention.py
        # from https://github.com/google-research/google-research/blob/master/performer/fast_attention/jax/fast_attention.py
    """

    """
    blocks = unstructured_blocks.reshape((nb_full_blocks, nb_columns, nb_columns))
    Q, _ = jnp.linalg.qr(blocks)
    final_matrix = Q.transpose((0, 2, 1)).reshape(-1, nb_columns)
    """

    # scales matrix back after QR
    if scaling == 0:
        multiplier = jnp.linalg.norm(
            random.normal(key, (nb_rows, nb_columns)), axis=1)
    elif scaling == 1:
        # this is supposed to cancel out the 1 / num_features?
        multiplier = jnp.sqrt(float(nb_columns)) * jnp.ones((nb_rows))
    elif scaling == 2:
        # use this since we are not normalizing by 1 / num_features.
        multiplier = jnp.ones((nb_rows,))
    else:
        raise ValueError('Scaling must be one of {0, 1, 2}. Was %s' % scaling)

    return multiplier[:,None] * final_matrix
    #return jnp.matmul(jnp.diag(multiplier), final_matrix)

# input: random samples `unstructured_blocks`
get_2d_arrays = jax.jit(jax.vmap(
    functools.partial(get_2d_array, scaling=0),
))

def random_projection(num_features, original_dim, key, bsz=None):
    shape = ((num_features, original_dim)
        if bsz is None
        else (bsz, num_features, original_dim)
    )
    return random.normal(key, shape)

# tests
def attn(q, k):
    log_pots = q @ k.T
    return jax.nn.softmax(log_pots), log_pots

lmm = jax.jit(lambda x,y: lse(x[:,None,:] + y[None,:,:], -1))

def rff_attn(q, k, projection_matrix, eps=0):
    kernel_cons = nonnegative_softmax_kernel_feature_creator
    log_phi_q = kernel_cons(
        q, projection_matrix, is_query=True, eps=eps,
        #normalize_data=True,
        normalize_data=False,
    )
    log_phi_k = kernel_cons(
        k, projection_matrix, is_query=False, eps=eps,
        #normalize_data=True,
        normalize_data=False,
    )
    log_pots_hat = log_phi_q[:,None,:] + log_phi_k[None,:,:]
    # average
    log_pots = lmm(log_phi_q, log_phi_k) - math.log(k.shape[0])
    return jnp.exp(log_pots - lse(log_pots, -1, keepdims=True)), log_pots_hat

rffa = jax.jit(rff_attn)

def rff_attn0(q, k, projection_matrix):
    kernel_cons = nonnegative_softmax_kernel_feature_creator0
    phi_q = kernel_cons(
        q, projection_matrix, (0,), None, is_query=True, eps=0,
        #normalize_data=True,
        normalize_data=False,
    )
    phi_k = kernel_cons(
        k, projection_matrix, (0,), None, is_query=False, eps=0,
        #normalize_data=True,
        normalize_data=False,
    )
    uprobs = phi_q @ phi_k.T
    return uprobs / uprobs.sum(-1, keepdims=True)

rffa0 = jax.jit(rff_attn0)

def relu_rff_attn0(q, k, projection_matrix):
    kernel_cons = relu_nonnegative_softmax_kernel_feature_creator0
    phi_q = kernel_cons(
        q, projection_matrix, (0,), None, is_query=True, eps=.0001,
        #normalize_data=True,
        normalize_data=False,
    )
    phi_k = kernel_cons(
        k, projection_matrix, (0,), None, is_query=False, eps=.0001,
        #normalize_data=True,
        normalize_data=False,
    )
    uprobs = phi_q @ phi_k.T
    #import pdb; pdb.set_trace()
    return uprobs / uprobs.sum(-1, keepdims=True), None

def relu_rff_attn(q, k, projection_matrix):
    kernel_cons = relu_nonnegative_softmax_kernel_feature_creator
    log_phi_q = kernel_cons(
        q, projection_matrix, (0,), None, is_query=True, eps=.0001,
        #normalize_data=True,
        normalize_data=False,
    )
    log_phi_k = kernel_cons(
        k, projection_matrix, (0,), None, is_query=False, eps=.0001,
        #normalize_data=True,
        normalize_data=False,
    )
    log_pots_hat = log_phi_q[:,None,:] + log_phi_k[None,:,:]
    # average
    log_pots = lmm(log_phi_q, log_phi_k)
    return jnp.exp(log_pots - lse(log_pots, -1, keepdims=True)), log_pots_hat

def exp_rff_attn(q, k, projection_matrix):
    kernel_cons = exp_nonnegative_softmax_kernel_feature_creator
    log_phi_q = kernel_cons(
        q, projection_matrix, (0,), None, is_query=True, eps=.0001,
        #normalize_data=True,
        normalize_data=False,
    )
    log_phi_k = kernel_cons(
        k, projection_matrix, (0,), None, is_query=False, eps=.0001,
        #normalize_data=True,
        normalize_data=False,
    )
    log_pots_hat = log_phi_q[:,None,:] + log_phi_k[None,:,:]
    # average
    log_pots = lmm(log_phi_q, log_phi_k)
    return jnp.exp(log_pots - lse(log_pots, -1, keepdims=True)), log_pots_hat

def sincos_rff_attn0(q, k, projection_matrix):
    kernel_cons = sincos_nonnegative_softmax_kernel_feature_creator0
    phi_q = kernel_cons(
        q, projection_matrix, (0,), None, is_query=True, eps=.0001,
        #normalize_data=True,
        normalize_data=False,
    )
    phi_k = kernel_cons(
        k, projection_matrix, (0,), None, is_query=False, eps=.0001,
        #normalize_data=True,
        normalize_data=False,
    )
    uprobs = phi_q @ phi_k.T
    #import pdb; pdb.set_trace()
    return uprobs / uprobs.sum(-1, keepdims=True), None


def kl(p, q):
    e_ratio = p * (jnp.log(p) - jnp.log(q))
    #e_ratio = jax.ops.index_update(e_ratio, p == 0, 0)
    return e_ratio.sum(-1)

def train(q, k, scale, proj, true_attn, L_dL, proj_fn,
    alpha, num_iters, key, sample=True,
    post_renorm=False,
):
    losses = onp.zeros((num_iters,))
    grads = onp.zeros((num_iters,))
    for i in range(num_iters):
        if sample:
            key, key_sample = jax.random.split(key)
        else:
            key_sample = key
        projection_matrix = proj_fn(key_sample)
        kl_val, (dq, dk) = L_dL(q, k, projection_matrix, true_attn)
        q -= alpha * dq
        k -= alpha * dk
        losses[i] = kl_val
        grads[i] += norm(dq) ** 2
        grads[i] += norm(dk) ** 2

        if post_renorm:
            q = renorm(q)
            k = renorm(k)

    return losses, grads, q, k, scale, projection_matrix

def train_proj(q, k, scale, proj,
    true_attn,
    L_dL,
    proj_fn_unused,
    alpha, num_iters, key, sample,
    post_renorm = False,
):
    losses = onp.zeros((num_iters,))
    grads = onp.zeros((num_iters,))
    for i in range(num_iters):
        kl_val, (dq, dk, dscale, dproj) = L_dL(q, k, scale, proj, true_attn)

        """
        # dbg
        ra, _ = relu_rff_attn0(q, k, proj)
        print(f"kl {kl_val}, attnmin {ra.min()}")
        if ra.min() < 0:
            import pdb; pdb.set_trace()
        if jnp.isinf(kl_val) or jnp.isnan(kl_val):
            import pdb; pdb.set_trace()
        #/dbg
        """
        
        q -= alpha * dq
        k -= alpha * dk
        if dscale is not None:
            scale -= alpha * dscale
        if dproj is not None:
            proj -= alpha * dproj

        losses[i] = kl_val

        grads[i] += norm(dq) ** 2
        grads[i] += norm(dk) ** 2
        if dscale is not None:
            grads[i] += norm(dscale) ** 2
        if dproj is not None:
            grads[i] += norm(dproj) ** 2

        #import pdb; pdb.set_trace()
        if post_renorm:
            q = renorm(q)
            k = renorm(k)

        #import pdb; pdb.set_trace()
        #print(f"grad {grads[i]}")
    #import pdb; pdb.set_trace()
    return losses, grads, q, k, scale, proj

