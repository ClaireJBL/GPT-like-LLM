import torch
import torch.nn as nn

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your     (x^1)
    [0.55, 0.87, 0.66], # journey  (x^2)
    [0.57, 0.85, 0.64], # starts (x^3)
    [0.22, 0.58, 0.33], # with (x^4)
    [0.77, 0.25, 0.10], # one (x^5)
    [0.05, 0.80, 0.55]] # step (x^6)
)

# this class is derived from nn.Module, which is a fundamental building block of PyTorch models that provides necessary functionalities for model layer creation and management
class SelfAttention_v1(nn.Module):
    # this metod initializes trainable weight matrices for queries, keys, and values, each transforming the input dimension d_in to d_out
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    # this method is to compute the attention scores by multiplying queries and keys, normalizing these scores using softmax
    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec


d_in = inputs.shape[1]  # the input embedding size, d = 3
d_out = 2 # the output embedding size, d_out = 2

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)

# instead of manually implementing nn.Parameter(torch.ran...), nn.Linear has an optimized weight initialization scheme, contributing to more stable and effective model training
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)    

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)

#compute the attention weights using the softmax function
queries = sa_v2.W_query(inputs) # reuses the query and key weight matrices of the SelfAttention_v2 object from the previous section for convenience
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

#use tril function to create a mask where the values above the diagonal are zero
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))

#multiply the mask with the attention weights to zero-out the values above the diagonal
masked_simple = attn_weights*mask_simple

#renormalize the attention weights to sum up to 1 again in each row
row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums

# a more efficient way to obtain the masked attention weight matrix is to mask the attention scores with negative infinity values before applying the softmax function
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)

# apply the softmax function to these masked results
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)

#dropout
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5) # dropout rate of 50%
print(dropout(attn_weights))
