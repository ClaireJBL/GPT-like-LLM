import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

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


torch.manual_seed(123)

X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
])
y_train = torch.tensor([0, 0, 0, 1, 1])
X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6],
])
y_test = torch.tensor([0, 1])

class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y
    
    #instructions for retrieving exactly one data record and the corresponding label
    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y
    
    #instructions for returning the total length of the dataset
    def __len__(self):
        return self.labels.shape[0] 
    
train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)


train_loader = DataLoader(
    dataset=train_ds, #The ToyDataset instance created earlier serves as input to the data loader
    batch_size=2,
    shuffle=True, #whether or not to shuffle the data
    num_workers=0, #the number of background processes
    drop_last=True
)

test_loader = DataLoader(
    dataset=test_ds,
    batch_size=2,
    shuffle=False, #not necessary to shuffle a test dataset
    num_workers=0
)

model = NeuralNetwork(num_inputs=2, num_outputs=2) #the dataset has two features and two classes


device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
 #defines a device variable that defaults to a GPU
model = model.to(device) #transfer the model onto the GPU

optimizer = torch.optim.SGD(
    model.parameters(), lr=0.5 #The optimizer needs to know which parameters to optimize.
)

num_epochs = 3
for epoch in range(num_epochs):

    model.train()

    for batch_idx, (features, labels) in enumerate(train_loader):
        features, labels = features.to(device),labels.to(device) #transfers the data onto the GPU
        logits = model(features)

        loss = F.cross_entropy(logits, labels) 

        optimizer.zero_grad() #Sets the gradients from the previous round to 0 to prevent unintended gradient accumulation
        loss.backward() #Computes the gradients of the loss given the model parameters
        optimizer.step() #The optimizer uses the gradients to update the model parameters

        ### LOGGING
        # print (f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
        #         f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
        #         f" | Train Loss: {loss:.2f}")

model.eval()
with torch.no_grad():
    outputs = model(X_train)
torch.set_printoptions(sci_mode=False) #it is used to make the outputs more legible
#probas = torch.softmax(outputs, dim=1) #the class membership probabilities using softmax
#predictions = torch.argmax(probas, dim=1) #returns the index position of the highest value in each row if we set dim=1 (dim=0 will return highest value in each column)
#print(predictions)

predictions = torch.argmax(outputs, dim=1)
#print(predictions)

#print(torch.sum(predictions == y_train)) #count the number of corret predictions

### COMPUTE PREDICTION ACCURACY
def compute_accuracy(model, dataloader):

    model = model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):

        with torch.no_grad():
            logits = model(features) 
        
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions #returns a tensor of True/False values depending on whether the labels match
        correct += torch.sum(compare) #The sum operation counts the number of True values
        total_examples += len(compare) 

    return (correct / total_examples).item() #The fraction of correct prediction, a value between 0 and 1. .item() returns the value of the tensor as a Python float.

#print (compute_accuracy(model, train_loader))
    
torch.save(model.state_dict(), "model.pth") #save the model