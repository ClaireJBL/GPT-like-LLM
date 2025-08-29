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
d_in = inputs.shape[1]  # the input embedding size, d = 3
d_out = 2 # the output embedding size, d_out = 2

batch = torch.stack((inputs, inputs), dim=0) # two inputs with six tokens each; each token has embedding dimension 3

#this class is similar to the SelfAttention class we implemented earlier, except that we added the dropout and causal mask components
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)  # Compared to the previous SelfAttention_v1 class, we added a dropout layer
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1) # the register bugger call is also a new addition, avoiding device mismatch errors
        ) 

    def forward(self, x):
        b, num_tokens, d_in = x.shape 
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2) # we transpose dimension 1 and 2, keeping the batch dimension at the first position (0)
        attn_scores.masked_fill_(  # In PyTorch, operations with a trailing underscore are performed in-place, avoiding unnecessary memory copies
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        context_vec = attn_weights @ values
        return context_vec
    

torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("context_vecs.shape:", context_vecs.shape)