import torch
import torch.nn as nn
import math

def nonnegative_softmax_kernel_feature_creator(
        data: torch.Tensor,
        projection_matrix: torch.Tensor,
        is_query: bool,
        eps: float=0.0001):
    """
    Constructs nonnegative kernel features for fast softmax attention.
    Args:
      data: input for which features are computes
      projection_matrix: random matrix used to compute features
      is_query: predicate indicating whether input data corresponds to queries or
        keys
      eps: numerical stabilizer.
    Returns:
      Random features for fast softmax attention.
    """

    ratio = 1.0 / math.sqrt(projection_matrix.shape[0])

    bsz = data.size(0)
   
    projection = projection_matrix.unsqueeze(0).expand(bsz, -1, -1)

    # Compute wx
    # data:       bsz, len, D
    # projection: bsz, D, #features
    data_dash = torch.bmm(
        data,
        projection
    ) # bsz, len, #features

    # Compute ||x||^2/2
    diag_data = torch.square(data)
    diag_data = torch.sum(diag_data, -1) # (bsz, len) ||x||^2
    diag_data = diag_data / 2.0
    diag_data = diag_data.unsqueeze(-1) # bsz, len, 1

    # Compute exp(wx - ||x||^2/2)  
    # (Lemma 1, SM(x, y) = E_{w~N(0,I)} exp(wx - ||x||^2/2) exp(wy - ||y||^2/2))
    if is_query:
        # for each query, we can independently scale to avoid overflows
        data_dash = ratio * (
            torch.exp(data_dash - diag_data -
                    torch.max(data_dash, dim=-1, keepdim=True)[0]) + eps)
    else:
        # for keys, we need to use the same normalizer to avoid overflows
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(data_dash)) + eps)

    return data_dash

kernel = nonnegative_softmax_kernel_feature_creator

def kl(attn_q, attn_p):
  """computes KL(q||p)"""
  return (-attn_q * attn_p.log() + attn_q * attn_q.log()).sum(-1)

if __name__ == "__main__":
    temperature_sqrt = 1.5

    torch.manual_seed(1234)
    D = 16 # D, queires and keys are vectors of size D
    query_len = 4
    key_len = 4
    num_features = 9999 # the number of projections we use, this is r in the paper. The more the better.

    query = torch.randn(1, query_len, D) / temperature_sqrt
    key = torch.randn(1, key_len, D) / temperature_sqrt


    true_values = torch.bmm(query, key.transpose(-1, -2) ).exp()
    true_attn = true_values / true_values.sum(-1, keepdim=True)
    print ('true attn:', true_attn)

    torch.manual_seed(5678)

    projection_matrix = torch.randn(D, num_features)
    query_features = kernel(query, projection_matrix, is_query=True, eps=0)
    key_features = kernel(key, projection_matrix, is_query=False, eps=0)

    values = torch.bmm(query_features, key_features.transpose(-1, -2))
    attn = values / values.sum(-1, keepdim=True)
    print ('linear attn:', attn)
    print ('kl:', kl(attn, true_attn))

    def get_2d_array(nb_rows, nb_columns, scaling=0):
      nb_full_blocks = int(nb_rows / nb_columns)
      block_list = []
      #rng = self.key
      for _ in range(nb_full_blocks):
        #rng, rng_input = jax.random.split(rng)
        unstructured_block = torch.randn(nb_columns, nb_columns)
        q, _ = torch.qr(unstructured_block)
        q = q.T
        block_list.append(q)
      remaining_rows = nb_rows - nb_full_blocks * nb_columns
      if remaining_rows > 0:
        unstructured_block = torch.randn(nb_columns, nb_columns)
        q, _ = torch.qr(unstructured_block)
        q = q.T
        block_list.append(q[0:remaining_rows])
      final_matrix = torch.cat(block_list, 0)
      #print (final_matrix.size())

      if scaling == 0:
        multiplier = torch.norm(
            torch.randn(nb_rows, nb_columns), dim=-1).view(-1, 1)
      elif scaling == 1:
        multiplier = torch.sqrt(float(nb_columns)) * torch.ones((nb_rows))
      else:
        raise ValueError('Scaling must be one of {0, 1}. Was %s' % scaling)

      return multiplier * final_matrix


    num_samples = 100
    num_features = 256
    samples = []
    for i in range(num_samples):
      projection_matrix = get_2d_array(num_features, D).transpose(0,1)
      query_features = kernel(query, projection_matrix, is_query=True, eps=0)
      key_features = kernel(key, projection_matrix, is_query=False, eps=0)

      values = torch.bmm(query_features, key_features.transpose(-1, -2))
      attn = values / values.sum(-1, keepdim=True)
      samples.append(attn)

    sample = torch.stack(samples)
    print(f"num_features {num_features}, D {D}")
    print("variance")
    print(sample.var(0))
    print("mean")
    print(sample.mean(0))
    print("true")
    print(true_attn)


