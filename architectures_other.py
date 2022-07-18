"""
Author: Mohamed Abdelaziz
Matr.Nr.: K12137202
Exercise 5
"""

import torch
from torch.nn import *


# Simple CNN  [Base Results]
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


# Dilated Block + Pooling  [Similar Results]
class DilatedBlock(Module):

    def __init__(self, n_in_channels=3, n_hidden_layers=1, n_kernels=32, kernel_size=3, out_chan=3, dilation=2, activ=ReLU(), pool=False):
        super(DilatedBlock, self).__init__()

        layers = []
        for i in range(n_hidden_layers):
            layers.append(Conv2d(n_in_channels, n_kernels, kernel_size, padding=int(dilation*(kernel_size - 1)/2), dilation=dilation))
            layers.append(activ)
            n_in_channels = n_kernels

        layers.append(Conv2d(n_kernels, 3, kernel_size, padding=int(kernel_size / 2)))
        if pool:
            if pool == 'avg':
                layers.append(AvgPool2d(kernel_size, stride=1, padding=int((kernel_size - 1)/2)))
            else:
                layers.append(MaxPool2d(kernel_size, stride=1, padding=int(dilation*(kernel_size - 1)/2)))
        layers.append(ReLU())
        self.output = Sequential(*layers)

    def forward(self, x):
        return self.output(x)


# AutoEncoder  [Worse Results]
class EncDecBlock(Module):

    def __init__(self, n_in_channels=3, n_hidden_layers=1, n_kernels=32, kernel_size=3, out_chan=3, activ=ReLU()):
        super(EncDecBlock, self).__init__()

        layers = []

        layers.append(Conv2d(int(n_in_channels), n_kernels, kernel_size, padding=int(kernel_size / 2)))
        layers.append(activ)

        n_in_channels = n_kernels
        # Encoder
        for i in range(n_hidden_layers):
            layers.append(Conv2d(int(n_in_channels), int(n_in_channels*2), kernel_size, padding=int(kernel_size / 2)))
            layers.append(activ)
            n_in_channels *= 2

        # Processor -> Should be Attention layer
        # layers.append(SimpleCNN(n_in_channels=n_in_channels, n_hidden_layers=n_hidden_layers, kernel_size=kernel_size, n_out_channels=n_in_channels))

        # Decoder
        for i in range(n_hidden_layers):
            layers.append(Conv2d(int(n_in_channels), int(n_in_channels/2), kernel_size, padding=int(kernel_size / 2)))
            layers.append(activ)
            n_in_channels /= 2

        layers.append(Conv2d(int(n_in_channels), out_chan, kernel_size, padding=int(kernel_size / 2)))
        layers.append(ReLU())
        self.output = Sequential(*layers)

    def forward(self, x):
        return self.output(x)


# SimpleCNN + Simple AutoEncoder  [Similar Results]
class Deblur(Module):
    def __init__(self, n_kernels):
        # Simple AutoEnc
        super(Deblur, self).__init__()
        self.c1 = Conv2d(3, n_kernels, kernel_size=9, padding=2)
        self.r1 = ReLU()
        self.c2 = Conv2d(n_kernels, int(n_kernels/2), kernel_size=1, padding=2)
        self.r2 = ReLU()
        self.c3 = Conv2d(int(n_kernels/2), 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.r1(self.c1(x))
        x = self.r2(self.c2(x))
        return self.c3(x)

class Inpainter(Module):
    """docstring for Inpainter."""

    def __init__(self, n_in_channels=3, n_hidden_layers=1, n_kernels=32, kernel_size=3, out_chan=3):
        super(Inpainter, self).__init__()
        self.block = SimpleCNN(n_in_channels=n_in_channels, n_hidden_layers=n_hidden_layers, n_kernels=n_kernels, kernel_size=kernel_size)
        self.deblur = Deblur(n_kernels)

    def forward(self, x):
        return self.deblur(self.block(x))




# Different Kernel Sizes [Similar Results]
class VariantCNN(Module):
    def __init__(self, n_in_channels=3, n_hidden_layers=3, n_kernels=32, kernel_size=7, n_out_channels=3):
        """Simple CNN with `n_hidden_layers`, `n_kernels`, and `kernel_size` as hyperparameters"""
        super().__init__()

        cnn = []
        for i in range(n_hidden_layers):
            cnn.append(Conv2d(n_in_channels, n_kernels, 7, padding=3))
            cnn.append(ReLU())
            n_in_channels = n_kernels

        for i in range(n_hidden_layers):
            cnn.append(Conv2d(n_in_channels, n_kernels, 5, padding=2))
            cnn.append(ReLU())
            n_in_channels = n_kernels

        for i in range(n_hidden_layers):
            cnn.append(Conv2d(n_in_channels, n_kernels, 3, padding=1))
            cnn.append(ReLU())
            n_in_channels = n_kernels

        cnn.append(Conv2d(in_channels=n_in_channels, out_channels=n_out_channels, kernel_size=1, padding=0))
        # cnn.append(ReLU())
        self.output = Sequential(*cnn)

    def forward(self, x):
        return self.output(x)
