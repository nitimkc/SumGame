# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

# In EGG, the game designer must implement the core functionality of the Sender and Receiver agents. These are then
# embedded in wrappers that are used to train them to play Gumbel-Softmax- or Reinforce-optimized games. The core
# Sender must take the input and produce a hidden representation that is then used by the wrapper to initialize
# the RNN or other module that will generate the message. The core Receiver expects a hidden representation
# generated by the message-processing wrapper, plus possibly other game-specific input, and it must generate the
# game-specific output.

# The Receiver class implements the core Receiver agent for the summing game. This is simply a linear layer
# that takes as input the vector generated by the message-decoding RNN in the wrapper (x in the forward method) and
# produces an output of n_features dimensionality, to be interpreted as a one-hot representation of the sum as
# attribute-value vector
class Receiver(nn.Module):
    def __init__(self, n_hidden, n_features):
        super(Receiver, self).__init__()
        self.output = nn.Linear(n_hidden, n_features)

    def forward(self, x, _input, _aux_input):
        return self.output(x)


# The Sender class implements the core Sender agent common to both games: it gets the input target vector and produces 
# a hidden layer that will initialize the message producing RNN
class Sender(nn.Module):
    def __init__(self, n_hidden, n_features):
        super(Sender, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)

    def forward(self, x, _aux_input):
        return self.fc1(x)
        # here, it might make sense to add a non-linearity, such as tanh
