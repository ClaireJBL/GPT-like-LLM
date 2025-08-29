import torch
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your     (x^1)
    [0.55, 0.87, 0.66], # journey  (x^2)
    [0.57, 0.85, 0.64], # starts (x^3)
    [0.22, 0.58, 0.33], # with (x^4)
    [0.77, 0.25, 0.10], # one (x^5)
    [0.05, 0.80, 0.55]] # step (x^6)
)

# query = inputs[1]
# attn_scores_2 = torch.empty(inputs.shape[0]) # the second input token serves as the query
# for i, x_i in enumerate(inputs):
#     attn_scores_2[i] = torch.dot(x_i, query) #the dot product is a measure of similarity because it quantifies how closely two vectors are aligned: a higher dot product indicates a greater degree of alignment


# following are normalization step, inpractice, more common to use the softmax function for normalization
# attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
# print("Attention weights:", attn_weights_2_tmp)
# print("Sum:", attn_weights_2_tmp.sum())

# following is using the PyTorch softmax funtion
# attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
# # print("Attention weights:", attn_weights_2)
# # print("Sum:", attn_weights_2.sum())

# # calculating the context vector z(x^2) by multiplying the embedded input tokens with the corresponding attention weights and then summing the resulting vectors
# query = inputs[1]  # the second input token is the query
# context_vec_2 = torch.zeros(query.shape)
# for i,x_i in enumerate(inputs):
#     context_vec_2 += attn_weights_2[i]*x_i
# print(context_vec_2)


#  ** Compute attention weights for all input tokens **
# attn_scores = torch.empty(6, 6)
# for i, x_i in enumerate(inputs):
#     for j, x_j in enumerate(inputs):
#         attn_scores[i, j] = torch.dot(x_i, x_j)

# instead of using for loop, we can just use matrix multiplication
attn_scores = inputs @ inputs.T

# normalize
attn_weights = torch.softmax(attn_scores, dim=-1) #dim=-1 instructs the softmax funtion to apply the normalization along the last dimension of the attn_scores tensor

# compute all context vectors
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)

