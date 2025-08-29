import torch
import torch.nn.functional as F #this import statement is a common convention in PyTorch to prevent long lines of code
from torch.autograd import grad

y = torch.tensor([1.0]) #true lable
x1 = torch.tensor([1.1]) #input feature
w1 = torch.tensor([2.2], requires_grad=True) #weight parameter
b = torch.tensor([0.0], requires_grad=True) #bias unit

z = x1 * w1 + b #net input
a = torch.sigmoid(z) #activaton and output

loss = F.binary_cross_entropy(a, y)

#By default, PyTorch destroys the computation graph after calculating the gradients to free memory.
#However, since we will reuse this computation graph shortly, we set retain_graph=True so it stays in the memory.
grad_L_w1 = grad(loss, w1, retain_graph=True)
grad_L_b = grad(loss, b, retain_graph=True)
#we are using the grad function mnually, but we can also use the high-level tool .bakcword to automate the process
print(grad_L_w1)
print(grad_L_b)
#or
loss.backward()
print(w1.grad)
print(b.grad)
