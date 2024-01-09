import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1)

def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=1)


class Net(nn.Module):
    def __init__(self, board_size, action_size, num_resBlocks=20, num_hidden=128):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initial convolution 
        self.startBlock = nn.Sequential(
            conv3x3(3, num_hidden),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        
        # Loop of all 20 Residual Layers
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )
        
        
        # Outputs expected value of the state 
        self.valueHead = nn.Sequential(
            conv1x1(num_hidden, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),

            nn.Linear(in_features=1, out_features=num_hidden),
            nn.ReLU(),

            nn.Linear(in_features=num_hidden, out_features=1),
            nn.nn.Tanh()
        )

    
        # Outputs the probabilities of each possible action 
        self.policyHead = nn.Sequential(
            conv1x1(num_hidden, 2),
            nn.BatchNorm2d(2),
            nn.ReLU(),

            nn.Linear(in_features=1, out_features=(action_size)),
            nn.Softmax(dim=1)
        )


        self.to(self.device)

    def forward(self, x):
        x = self.startBlock(x)

        for resBlock in self.backBone:
            x = resBlock(x)

        policy = self.policyHead(x)
        value = self.valueHead(x)

        return policy, value
    

class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = conv3x3(num_hidden, num_hidden)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = conv3x3(num_hidden, num_hidden)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Skip connections
        out += identity
        out = self.relu(out)

        return out