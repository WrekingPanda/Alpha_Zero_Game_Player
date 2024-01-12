import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Net(Module):
    def __init__(self, size, action_size, num_resBlocks, num_hidden):
        super().__init__()

        # Check if a GPU is available, otherwise use CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initial block: Convolution + Batch Normalization + ReLU
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=size, padding='same'),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        
        # Backbone with multiple residual blocks
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden, size) for i in range(num_resBlocks)]
        )
        
        # Policy head: Convolution + Batch Normalization + ReLU + Flatten + Linear + Softmax
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=size, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * size * size, action_size),
            nn.Softmax(dim=1)
        )
        
        # Value head: Convolution + Batch Normalization + ReLU + Flatten + Linear + Tanh
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, size, kernel_size=size, padding='same'),
            nn.BatchNorm2d(size),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(size * size * size, 1),
            nn.Tanh()
        )

        # Set the model to run on the selected device
        self.to(self.device)
    
    def forward(self, x):
        # Forward pass through the network
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value

class ResBlock(nn.Module):
    def __init__(self, num_hidden, size):
        super().__init__()
        # Residual block: Convolution + Batch Normalization + ReLU + Convolution + Batch Normalization
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=size, padding='same')
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=size, padding='same')
        self.bn2 = nn.BatchNorm2d(num_hidden)
        
    def forward(self, x):
        # Forward pass through the residual block
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x