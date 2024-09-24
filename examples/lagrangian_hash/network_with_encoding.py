from typing import Callable, Optional, List
from arrgh import arrgh

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import SplashEncoding
from .network import Network

try:
    import tinycudann as tcnn
except ImportError as e:
    print(
        f"Error: {e}! "
        "Please install tinycudann by: "
        "pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
    )
    exit()


class NetworkwithSplashEncoding(nn.Module):
    def __init__(
        self,
        base_resolution: int = 16,
        per_level_scale: int = 1.47,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        num_splashes: int = 4, 
        log2_hashmap_size: int = 17,
        splits: List[float] = [0.875, 0.9375],
        std_init_factor: float = 1.0,
        fixed_std: bool = False,
        decay_factor: int = 1,
        output_dim: int = 3,  # The number of output tensor channels.
        net_depth: int = 2,  # The depth of the MLP.
        net_width: int = 64,  # The width of the MLP.
        hidden_activation: str = "ReLU",
        output_activation: str = "None",
    ):
        super().__init__()

        self.encoding = SplashEncoding(base_resolution=base_resolution, per_level_scale=per_level_scale,
                                       n_levels=n_levels, n_features_per_level=n_features_per_level, 
                                       num_splashes=num_splashes, log2_hashmap_size=log2_hashmap_size,
                                       splits=splits, std_init_factor=std_init_factor, fixed_std=fixed_std, 
                                       decay_factor=decay_factor)
        
        input_dim = n_features_per_level * n_levels
        self.mlp = tcnn.Network(
                n_input_dims=input_dim,
                n_output_dims=output_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": hidden_activation,
                    "output_activation": output_activation,
                    "n_neurons": net_width,
                    "n_hidden_layers": net_depth,
                },
            )
        # self.mlp = Network(input_dim=input_dim, output_dim=output_dim, net_depth=net_depth, net_width=net_width,
        #                    hidden_init=hidden_init, hidden_activation=hidden_activation, output_init=output_init,
        #                    output_activation=output_activation, bias_enabled=bias_enabled, bias_init=bias_init)
        

    def forward(self, coords):
        encoding, gmm = self.encoding(coords)
        output = self.mlp(encoding)
        return output, gmm