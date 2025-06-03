import torch
import torch.nn as nn
import torch.nn.functional as F

#creating a model class that inherits nn.Module (neural network module)
class Model(nn.Module):
    #input layer (4 flower features) --> 
    # hidden layer 1 (number of neurons) --> 
    # hidden layer 2 (number of neurons) --> 
    # output (class of iris flower - 3 options)

    def __init__(self, input_features=4, h1=8, h2=8, output_features=3):
        super().__init__() #instantiate our nn.Module
        self.fc1 = nn.Linear(input_features, h1) #fc = fully connected
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, output_features)

    def forward(self, x):
        x = F.relu(self.fc1(x)) #rectified linear unit
        x = F.relu(self.fc2(x))
        x = F.relu(self.out(x))

        return x
    
#pick a manual seed for randomization
torch.manual_seed(41)
#create an instance of model
model = Model()