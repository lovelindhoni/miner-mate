import torch.nn as nn


class NeuralNetwork(nn.Module): # this is our model's neural network
    def __init__(self, input_size, hidden_size, output_size): #hidden size represents the no. of neurons in hidden layers
        super(NeuralNetwork, self).__init__() 
        # we have used three hidden layers
        self.layer1 = nn.Linear(input_size, hidden_size) 
        self.layer2 = nn.Linear(hidden_size, hidden_size) # the hidden layer
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU() # the activation function that intruduces non linearity by changing all the negative values to zero.
    
    def forward(self,x):
        output = self.layer1(x) # layer 1 with x(input data)
        output = self.relu(output) # activating relu
        output = self.layer2(output) # similarly for all layers
        output = self.relu(output)
        output = self.layer3(output)
        return output
