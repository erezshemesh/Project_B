import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim, bound=-1000):  # TODO: 22.03.23 bound=math.nan
        """
            Initialize the network and set up the layers.

            Parameters:
                in_dim - input dimensions as an int
                out_dim - output dimensions as an int
                bound - input
            Return:
                None
        """
        self.bound = bound
        super(FeedForwardNN, self).__init__()
        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)

    def forward(self, obs):
        """
            Runs a forward pass on the neural network.

            Parameters:
                obs - observation to pass as input
            Return:
                output - the output of our forward pass
        """
        # Convert observation to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        if self.bound != -1000:
            output = torch.tanh(self.layer3(activation2) / 1000)
            output = output.mul(self.bound)
        else:
            output = self.layer3(activation2)
        return output
