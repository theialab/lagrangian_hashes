"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(
        self,
        input_dim: int,  # The number of input tensor channels.
        output_dim: int = None,  # The number of output tensor channels.
        net_depth: int = 2,  # The depth of the MLP.
        net_width: int = 64,  # The width of the MLP.
        hidden_init: Callable = nn.init.xavier_uniform_,
        hidden_activation: Callable = nn.ReLU(),
        output_init: Optional[Callable] = nn.init.xavier_uniform_,
        output_activation: Optional[Callable] = nn.Identity(),
        bias_enabled: bool = True,
        bias_init: Callable = nn.init.zeros_,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net_depth = net_depth
        self.net_width = net_width
        self.hidden_init = hidden_init
        self.hidden_activation = hidden_activation
        self.output_init = output_init
        self.output_activation = output_activation
        self.bias_enabled = bias_enabled
        self.bias_init = bias_init

        self.hidden_layers = nn.ModuleList()
        in_features = self.input_dim
        for i in range(self.net_depth):
            self.hidden_layers.append(
                nn.Linear(in_features, self.net_width, bias=bias_enabled)
            )
            in_features = self.net_width
        self.output_layer = nn.Linear(
            in_features, self.output_dim, bias=bias_enabled
        )

        self.initialize()

    def initialize(self):
        def init_func_hidden(m):
            if isinstance(m, nn.Linear):
                if self.hidden_init is not None:
                    self.hidden_init(m.weight)
                if self.bias_enabled and self.bias_init is not None:
                    self.bias_init(m.bias)

        self.hidden_layers.apply(init_func_hidden)

        def init_func_output(m):
            if isinstance(m, nn.Linear):
                if self.output_init is not None:
                    self.output_init(m.weight)
                if self.bias_enabled and self.bias_init is not None:
                    self.bias_init(m.bias)

        self.output_layer.apply(init_func_output)

    def forward(self, x):
        for i in range(self.net_depth):
            x = self.hidden_layers[i](x)
            x = self.hidden_activation(x)
            
        x = self.output_layer(x)
        x = self.output_activation(x)
        return x