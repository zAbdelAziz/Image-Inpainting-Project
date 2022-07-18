"""
Author: Mohamed Abdelaziz
Matr.Nr.: K12137202
Exercise 5
"""

import torch
from torch.nn import *


# Simple CNN
class SimpleCNN(Module):
    def __init__(self, n_in_channels=3, n_hidden_layers=3, n_kernels=32, kernel_size=7, n_out_channels=3):
        """Simple CNN with `n_hidden_layers`, `n_kernels`, and `kernel_size` as hyperparameters"""
        super().__init__()

        cnn = []
        for i in range(n_hidden_layers):
            cnn.append(Conv2d(n_in_channels, n_kernels, kernel_size, padding=int(kernel_size / 2)))
            cnn.append(ReLU())
            n_in_channels = n_kernels

        cnn.append(Conv2d(in_channels=n_in_channels, out_channels=n_out_channels, kernel_size=kernel_size, padding=int(kernel_size / 2)))
        self.hidden_layers = Sequential(*cnn)

    def forward(self, x):
        return self.hidden_layers(x)
