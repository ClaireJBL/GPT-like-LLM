import torch
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your     (x^1)
    [0.55, 0.87, 0.66], # journey  (x^2)
    [0.57, 0.85, 0.64], # starts (x^3)
    [0.22, 0.58, 0.33], # with (x^4)
    [0.77, 0.25, 0.10], # one (x^5)
    [0.05, 0.80, 0.55]] # step (x^6)
)

x_2 = inputs[1] # the second input element
d_in = inputs.shape[1]  # the input embedding size, d = 3
d_out = 2 # the output embedding size, d_out = 2

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False) # if we were to use the weight matrices for model training, we would set requires_grad=True to update these matrices during model training


query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value

keys = inputs @ W_key
values = inputs @ W_value

attn_scores_2 = query_2 @ keys.T  # attention scores between all the vectors and the query vector

d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1) # the scaled attention weights are computed using the softmax funtion

#compute the context vector by combining all value vectors via the attention weights
context_vec_2 = attn_weights_2 @ values
print(context_vec_2)

# So far, all the above code only computed a single context vector, z(2), following will generalize the code to compute all the context vectors in the input sequence

