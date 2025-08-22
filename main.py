import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        # Always call the parent constructor first
        super().__init__()

        # 1. Define the layers (the "bricks")
        self.flatten = nn.Flatten() # Takes a [28, 28] tensor and makes it a [784] vector
        self.layer1 = nn.Linear(in_features=784, out_features=128) # A linear layer
        self.activation = nn.ReLU() # A non-linear activation function
        self.layer2 = nn.Linear(in_features=128, out_features=10) # The final output layer

    def forward(self, x):
        # 2. Define the data flow (the "instructions")
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

# Create an instance of our network and print it
model = SimpleNet()
print(model)
