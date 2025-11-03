import torch
import torch.nn as nn


class MyNN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(MyNN, self).__init__()
        layers = []

        prev_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1) # flatten x to bee=[28*28,batchsize=64]
        return self.network(x)