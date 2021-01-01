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

import numpy as onp

import matplotlib.pyplot as plt
import plotly.graph_objects as go

import streamlit as st

onp.set_printoptions(suppress=True, precision=2)

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
    ratio = 1.0 / jnp.sqrt(projection_matrix.shape[0])
    #ratio = 1.0
    #data_mod_shape = data.shape[0:len(batch_dims_t)] + projection_matrix.shape
    #data_thick_random_matrix = jnp.zeros(data_mod_shape) + projection_matrix

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

  #return data_dash - diag_data 

  #"""
  if is_query:
    last_dims_t = (len(data_dash.shape) - 1,)
    data_dash = ratio * (
        jnp.exp(data_dash - diag_data -
                jnp.max(data_dash, axis=last_dims_t, keepdims=True)) + eps)
  else:
    data_dash = ratio * (
        jnp.exp(data_dash - diag_data - jnp.max(data_dash)) + eps)

  return data_dash
  #"""

# garbage
class RandomMatrix(object):
  r"""Abstract class providing a method for constructing 2D random arrays.

  Class is responsible for constructing 2D random arrays.
  """

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def get_2d_array(self):
    raise NotImplementedError('Abstract method')


class GaussianUnstructuredRandomMatrix(RandomMatrix):

    def __init__(self, nb_rows, nb_columns, key):
        self.nb_rows = nb_rows
        self.nb_columns = nb_columns
        self.key = key

    def get_2d_array(self, bsz=None):
        if bsz is None:
            return random.normal(self.key, (self.nb_rows, self.nb_columns))
        else:
            return random.normal(self.key, (bsz, self.nb_rows, self.nb_columns))


class GaussianOrthogonalRandomMatrix(RandomMatrix):
  r"""Class providing a method to create Gaussian orthogonal matrix.

  Class is responsible for constructing 2D Gaussian orthogonal arrays.
  """

  def __init__(self, nb_rows, nb_columns, key, scaling=0):
    self.nb_rows = nb_rows
    self.nb_columns = nb_columns
    self.key = key
    self.scaling = scaling

  def get_2d_array(self, bsz=None):
    nb_full_blocks = int(self.nb_rows / self.nb_columns)
    block_list = []
    rng = self.key
    for _ in range(nb_full_blocks):
      rng, rng_input = jax.random.split(rng)
      unstructured_block = random.normal(rng_input,
                                         (self.nb_columns, self.nb_columns))
      q, _ = jnp.linalg.qr(unstructured_block)
      q = jnp.transpose(q)
      block_list.append(q)
    remaining_rows = self.nb_rows - nb_full_blocks * self.nb_columns
    if remaining_rows > 0:
      rng, rng_input = jax.random.split(rng)
      unstructured_block = random.normal(rng_input,
                                         (self.nb_columns, self.nb_columns))
      q, _ = jnp.linalg.qr(unstructured_block)
      q = jnp.transpose(q)
      block_list.append(q[0:remaining_rows])
    final_matrix = jnp.vstack(block_list)

    if self.scaling == 0:
      multiplier = jnp.linalg.norm(
          random.normal(self.key, (self.nb_rows, self.nb_columns)), axis=1)
    elif self.scaling == 1:
      multiplier = jnp.sqrt(float(self.nb_columns)) * jnp.ones((self.nb_rows))
    else:
      raise ValueError('Scaling must be one of {0, 1}. Was %s' % self._scaling)

    return jnp.matmul(jnp.diag(multiplier), final_matrix)

# /garbage

def get_2d_array(unstructured_blocks, key, scaling=0):
    nb_rows, nb_columns = unstructured_blocks.shape
    nb_full_blocks = int(nb_rows / nb_columns)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        raise ValueError("Assert nb_rows % nb_columns == 0 for simplicity")

    blocks = unstructured_blocks.reshape((nb_full_blocks, nb_columns, nb_columns))
    Q, _ = jnp.linalg.qr(blocks)
    final_matrix = Q.transpose((0, 2, 1)).reshape(-1, nb_columns)

    if scaling == 0:
        # what is this for?
        multiplier = jnp.linalg.norm(
            random.normal(key, (nb_rows, nb_columns)), axis=1)
    elif scaling == 1:
        multiplier = jnp.sqrt(float(nb_columns)) * jnp.ones((nb_rows))
    else:
        raise ValueError('Scaling must be one of {0, 1}. Was %s' % scaling)

    return jnp.matmul(jnp.diag(multiplier), final_matrix)

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

def rff_attn(q, k, projection_matrix):
    kernel_cons = nonnegative_softmax_kernel_feature_creator
    log_phi_q = kernel_cons(
        q, projection_matrix, is_query=True, eps=0, normalize_data=True)
    log_phi_k = kernel_cons(
        k, projection_matrix, is_query=False, eps=0, normalize_data=True)
    log_pots_hat = log_phi_q[:,None,:] + log_phi_k[None,:,:]
    # average
    log_pots = lmm(log_phi_q, log_phi_k) - math.log(k.shape[0])
    return jnp.exp(log_pots - lse(log_pots, -1, keepdims=True)), log_pots_hat

rffa = jax.jit(rff_attn)

def rff_attn0(q, k, projection_matrix):
    kernel_cons = nonnegative_softmax_kernel_feature_creator0
    phi_q = kernel_cons(
        q, projection_matrix, (0,), None, is_query=True, eps=0, normalize_data=True)
    phi_k = kernel_cons(
        k, projection_matrix, (0,), None, is_query=False, eps=0, normalize_data=True)
    uprobs = phi_q @ phi_k.T
    return uprobs / uprobs.sum(-1, keepdims=True)

rffa0 = jax.jit(rff_attn0)

def kl(p, q):
    e_ratio = p * (jnp.log(p) - jnp.log(q))
    #e_ratio = jax.ops.index_update(e_ratio, p == 0, 0)
    return e_ratio.sum(-1)

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)

    key, key1, key2 = jax.random.split(key, 3)
    qk_dim = 8
    T = 4
    q = jax.random.normal(key1, (T, qk_dim)) 
    k = jax.random.normal(key2, (T, qk_dim))
    qu, ku = q.copy(), k.copy()
    qs, ks = q.copy(), k.copy()

    num_features = 256
    #num_features = 32000

    st.markdown("## Categorical approximation")
    st.write("We start by trying to approximate a small-dimensional categorical distribution.")

    st.write("We use the RFF approximation with")
    st.write(f"num_features = {num_features}")

    unstructured = GaussianUnstructuredRandomMatrix(num_features, qk_dim, key).get_2d_array()
    structured = GaussianOrthogonalRandomMatrix(num_features, qk_dim, key).get_2d_array()

    attn_dist, logits = attn(q, k)
    #projection_matrix = random.normal(key, (num_features, qk_dim))
    projection_matrix = random_projection(num_features, qk_dim, key)
    rff_unstruct_attn_dist, unstruct_logits_hat = rff_attn(qu, ku, projection_matrix)

    rff_unstruct_attn_dist0 = rff_attn0(qu, ku, projection_matrix)


    _, newkey = jax.random.split(key)
    samples = []
    samples0 = []
    for i in range(100):
        newkey, key1 = jax.random.split(newkey)
        unstructured = GaussianUnstructuredRandomMatrix(num_features, qk_dim, key1).get_2d_array()
        a = rff_attn0(qu, ku, unstructured)
        b, _ = rff_attn(qu, ku, unstructured)
        samples0.append(a)
        samples.append(b)
    samples = onp.stack(samples)
    samples0 = onp.stack(samples0)
    #import pdb; pdb.set_trace()

    projection_matrix0 = structured
    rff_struct_attn_dist0, struct_logits_hat0 = rff_attn(qs, ks, projection_matrix0)
    projection_matrix = get_2d_array(projection_matrix, key)
    rff_struct_attn_dist, struct_logits_hat = rff_attn(qs, ks, projection_matrix)

    _, key1 = random.split(key)
    projection_matrix1 = get_2d_array(projection_matrix, key1)
    rff_struct_attn_dist1, struct_logits_hat1 = rff_attn(qs, ks, projection_matrix1)

    def write(x):
        st.dataframe(jnp.asarray(x))

    st.write("Given a set of query and key embeddings, we compare the RFF approximation to the true distributions")
    st.write("kl(attn, unstruct_attn)")
    write(kl(attn_dist, rff_unstruct_attn_dist))
    st.write("kl(attn, struct_attn)")
    write(kl(attn_dist, rff_struct_attn_dist))
    st.write("kl(attn, struct_attn0)")
    write(kl(attn_dist, rff_struct_attn_dist0))

    st.write("The exponentiated logits, which the RFF approximates")
    st.write("attn exp(logits)")
    write(jnp.exp(logits))
    st.write("unstruct attn exp(logits)")
    write(jnp.exp(unstruct_logits_hat).mean(-1))
    st.write("struct attn exp(logits)")
    write(jnp.exp(struct_logits_hat).mean(-1))
    st.write("struct attn exp(logits0)")
    write(jnp.exp(struct_logits_hat0).mean(-1))

    st.write("We also measure variance in a given exp kernel approximation")
    st.write("unstruct attn var exp(logits)")
    write(jnp.exp(unstruct_logits_hat).var(-1))
    st.write("struct attn var exp(logits)")
    write(jnp.exp(struct_logits_hat).var(-1))
    st.write("struct attn var exp(logits0)")
    write(jnp.exp(struct_logits_hat0).var(-1))


    # check mean and variance of original implementation
    NUM_ITERS = 100
    attn_dist, logits = attn(q, k)
    attns = []
    attns0 = []
    for i in range(NUM_ITERS):
        key, key1 = random.split(key)
        structured = GaussianOrthogonalRandomMatrix(num_features, qk_dim, key1).get_2d_array()

        projection_matrix0 = structured
        rff_struct_attn_dist0, _ = rffa(qs, ks, projection_matrix0)
        projection_matrix = get_2d_array(projection_matrix, key1)
        rff_struct_attn_dist, _ = rffa(qs, ks, projection_matrix)

        attns.append(onp.array(rff_struct_attn_dist))
        attns0.append(onp.array(rff_struct_attn_dist0))

    attns = onp.stack(attns)
    attns0 = onp.stack(attns0)
    import pdb; pdb.set_trace()

    st.markdown("## Learning an RFF Cat")
    # can we optimize to minimize variance + KL?
    def loss_u(q, k, attn_dist, projection_matrix):
        rff_unstruct_attn_dist, _ = rff_attn(q, k, projection_matrix)
        return kl(attn_dist, rff_unstruct_attn_dist).mean()

    L = jax.jit(jax.value_and_grad(loss_u, argnums=(0, 1)))

    """
    NUM_ITERS = 1000
    alpha = 1e-2
    losses = onp.zeros((NUM_ITERS,))
    for i in range(NUM_ITERS):
        key, key1 = jax.random.split(key, 2)
        unstructured_blocks = random.normal(key1, (num_features, qk_dim))
        projection_matrix = get_2d_array(unstructured_blocks, key1)
        projection_matrix0 = GaussianOrthogonalRandomMatrix(num_features, qk_dim, key1).get_2d_array()
        #import pdb; pdb.set_trace()
        kl_val, (dq, dk) = L(qu, ku, attn_dist, projection_matrix)
        qu -= alpha * dq
        ku -= alpha * dk
        losses[i] = kl_val

    fig = go.Figure(data=go.Scatter(
        x = onp.arange(NUM_ITERS),
        y = losses,
        mode = "markers",
    ))
    st.plotly_chart(fig, use_container_width=True)
    """


    # get covariance
    N = 2
    key, key1 = jax.random.split(key, 2)
    unstructured_blocks = random_projection(num_features, qk_dim, key1, bsz=N)
    keys = jax.random.split(key, N+1)
    key = keys[0]
    projection_matrices = get_2d_arrays(unstructured_blocks, keys[1:])
    rff_attns = jax.jit(jax.vmap(rff_attn, in_axes=(None,None,0))) # only broadcast projection_matrices

    attn_dists, logits_hat = rff_attns(qu, ku, projection_matrices)

    for i in range(5):
        key, key1 = jax.random.split(key, 2)
        projection_matrix = GaussianOrthogonalRandomMatrix(num_features, qk_dim, key1).get_2d_array()
        rff_attn_dist, logits_hat = rff_attn(qu, ku, projection_matrix)
        print(rff_attn_dist)
        print(jnp.exp(logits_hat).var(-1))
        #import pdb; pdb.set_trace()
