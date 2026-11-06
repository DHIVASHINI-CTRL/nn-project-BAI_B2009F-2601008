"""
model.py  -  Task 3: the neural network (designed by me, not from a lecture)

FrogMLP is a fully connected network (Multilayer Perceptron):

    22 MFCC features -> Dense(h1) -> ReLU -> [Dropout]
                     -> Dense(h2) -> ReLU -> [Dropout]
                     -> Dense(10 species)      (softmax is inside CrossEntropyLoss)

Why an MLP and not a CNN?
  The input is a row of 22 numbers per syllable, not an image. There is no
  spatial neighbourhood for convolution filters to exploit, so fully
  connected layers are the natural choice.
"""

import torch.nn as nn


class FrogMLP(nn.Module):
    def __init__(self, n_features=22, hidden=(64, 32), n_classes=10, dropout=0.0):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))  # weights: in_dim*h, biases: h
            layers.append(nn.ReLU())             # non-linearity, avoids saturation
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, n_classes))  # raw scores (logits)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def count_parameters(model):
    """Trainable parameters. For one dense layer: inputs*neurons + neurons."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def architecture_table(n_features=22, hidden=(64, 32), n_classes=10):
    """Rows of (layer, size, activation, parameters) for the report."""
    rows = [("Input", n_features, "-", 0)]
    in_dim = n_features
    for i, h in enumerate(hidden, 1):
        rows.append((f"Hidden {i}", h, "ReLU", in_dim * h + h))
        in_dim = h
    rows.append(("Output", n_classes, "Softmax (inside loss)", in_dim * n_classes + n_classes))
    return rows
