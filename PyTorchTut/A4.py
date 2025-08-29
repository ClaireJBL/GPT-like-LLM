# a classic multilayer perceptron with two hidden layers to illustrate a typical usage of the Module class
import torch

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs): #coding the number of inputs and outputs as variables allows us to reuse the same code for datasets with different numbers of features and classes
        super().__init__()

        self.layers = torch.nn.Sequential( #using sequential allows us the call the self.layers instead of calling each layer individaully in the forward method

            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30), #the Linear layer takes the number of input and output nodes as arguments
            torch.nn.ReLU(), #Nonlinear activation functions are placed between the hidden layer.

            # 2nd hidden layer
            torch.nn.Linear(30, 20), # the number of output nodes of one hidden layer has to match the number of inputs of the next layer.
            torch.nn.ReLU(), 

            # output layer
            torch.nn.Linear(20, num_outputs),
        )
    
    def forward(self, x):
        logits = self.layers(x)
        return logits # the outputs of the last layer are called logits
    
# instantiate a new neural network object
torch.manual_seed(123) #we can make the random number initialization reproducible
model = NeuralNetwork(50, 3)
#num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#print(model.layers[0].weight)
#similarly, could access te bias vector via model.layers[0].bias
#print("Total number of trainable model parameters:", num_params)

X = torch.rand((1, 50))
#with torch.no_grad(): #if we want to save memory and computation, not keep track of the gradients
#out = model(X)
# if we want to compute class-membership probabilities for our predictions, we have to call the softmax function
with torch.no_grad():
    out = torch.softmax(model(X), dim=1)
print(out)

