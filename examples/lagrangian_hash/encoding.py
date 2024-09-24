from typing import List
import math
import numpy as np
from arrgh import arrgh

import torch
import torch.nn as nn

import laghash.ops.grid as grid_ops


class SplashEncoding(nn.Module):
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
    ):
        """
        """
        super().__init__()

        self.num_lods = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.hashmap_size = int(2 ** log2_hashmap_size)
        splits.sort(reverse=True)
        self.splits = splits
        self.num_splashes = num_splashes
        self.decay_factor = decay_factor
        
        self.register_buffer("feat_begin_idxes", torch.zeros(self.num_lods+1, dtype=torch.int64))
        self.register_buffer("gau_begin_idxes", torch.zeros(self.num_lods+1, dtype=torch.int64))
        self.register_buffer("num_idxes", torch.zeros(self.num_lods, dtype=torch.int64))
        self.register_buffer("num_feats", torch.zeros(self.num_lods, dtype=torch.int64))
        self.register_buffer("num_gaus", torch.zeros(self.num_lods, dtype=torch.int64))

        self.resolutions = torch.zeros([self.num_lods], dtype=torch.int64)
        self.num_splashes = torch.zeros([self.num_lods], dtype=torch.int64)
        for i in range(self.num_lods):
            self.resolutions[i] = int(base_resolution * per_level_scale**i)
            for j in range(len(self.splits)):
                if i >= self.num_lods * self.splits[j]:
                    self.num_splashes[i] = num_splashes // (2**j)
                    assert(self.num_splashes[i] > 0, f"Num splashes is zero for LoD-{i}")
                    break
        print("# gaus in each LoD:", self.num_splashes)

        num_feats_so_far = 0
        num_gaus_so_far = 0
        for i in range(self.num_lods):
            max_index_level = self.hashmap_size
            num_index_level = int(self.resolutions[i] ** 3)
            num_index_level = min(num_index_level, max_index_level)

            num_gaus_level = int(num_index_level * self.num_splashes[i])
            num_feats_level = int(num_index_level * max(self.num_splashes[i], 1))

            self.feat_begin_idxes[i] = num_feats_so_far
            self.gau_begin_idxes[i] = num_gaus_so_far
            self.num_gaus[i] = num_gaus_level
            self.num_feats[i] = num_feats_level
            self.num_idxes[i] = num_index_level
            num_gaus_so_far += num_gaus_level
            num_feats_so_far += num_feats_level

        self.feat_begin_idxes[self.num_lods] = num_feats_so_far
        self.gau_begin_idxes[self.num_lods] = num_gaus_so_far

        r = 0.125
        self.total_feats = sum(self.num_feats)
        self.total_gaus = sum(self.num_gaus)
        self.feats = torch.randn(self.total_feats, self.n_features_per_level) * 1e-2
        self.feats = nn.Parameter(self.feats)
        self.means = torch.rand(self.total_gaus, 3)
        self.init_mean()
        self.means = nn.Parameter(self.means)
        self.stds = torch.ones(self.total_gaus, 1).cuda()
        self.init_std(std_init_factor)
        if not fixed_std:
            self.stds = torch.normal(r, 2e-2, size=(self.total_gaus, 1))
            self.stds = nn.Parameter(self.stds)
        print(f"Num grid features: {self.total_feats} and Num grid gaussians: {self.total_gaus}")
        # print(f"Num gaussians in each LoD:", self.num_gaus)


    def init_mean(self):
        N = self.total_gaus
        pts = np.random.randn(N, 3)
        r = np.sqrt(np.random.rand(N, 1))
        pts = pts / np.linalg.norm(pts, axis=1)[:, None] * r
        pts = pts * 0.25 + 0.5 # [0.25 ... 0.75]
        
        self.means = torch.tensor(pts, dtype=torch.float32)


    def init_std(self, std_init_factor):
        for lod in range(self.num_lods):
            if self.num_splashes[lod]:
                gau_size = std_init_factor * 2 / self.resolutions[lod]
                self.stds[self.gau_begin_idxes[lod]:self.gau_begin_idxes[lod+1]] *= gau_size


    def update_factor(self):
        self.stds = self.stds * self.decay_factor


    def get_feats(self, lod):
        """
        """
        feats = self.feats[self.feat_begin_idxes[lod]:self.feat_begin_idxes[lod+1]]
        feats = feats.view(-1, self.num_splashes[lod]+1, self.n_features_per_level)
        return feats


    def get_means(self, lod):
        """
        """
        if self.num_splashes[lod]:
            means = self.means[self.gau_begin_idxes[lod]:self.gau_begin_idxes[lod+1]]
            means = means.view(-1, self.num_splashes[lod], 3)
            return means
        else:
            return None


    def get_stds(self, lod):
        """
        """
        if self.num_splashes[lod]:
            stds = self.stds[self.gau_begin_idxes[lod]:self.gau_begin_idxes[lod+1]]
            stds = stds.view(-1, self.num_splashes[lod], 1)
            return stds
        else:
            return None
        

    def interpolate_cuda(self, coords):
        """Query multiscale features.

        Args:
            coords (torch.FloatTensor): coords of shape [batch, num_samples, 3] or [batch, 3]
                For some grid implementations, specifying num_samples may allow for slightly faster trilinear
                interpolation. HashGrid doesn't use this optimization, but allows this input type for compatability.
            lod_idx  (int): int specifying the index to ``active_lods``

        Returns:
            (torch.FloatTensor): interpolated features of shape
             [batch, num_samples, feature_dim] or [batch, feature_dim]
        """
        # Remember desired output shape
        output_shape = coords.shape[:-1]
        if coords.ndim == 3:                                          # flatten num_samples dim with batch for cuda call
            batch, num_samples, coords_dim = coords.shape             # batch x num_samples
            coords = coords.reshape(batch * num_samples, coords_dim)

        feats, gmm = grid_ops.interpolate(coords, self.feats, self.means, self.stds, 
                            self.feat_begin_idxes, self.gau_begin_idxes, 
                            self.log2_hashmap_size, self.resolutions, self.num_splashes)

        feats = feats.reshape(*output_shape, feats.shape[-1])
        return feats, gmm


    def forward(self, coords, lod_idx=None):
        # feats, gmm = self.interpolate(coords)

        feats, gmm = self.interpolate_cuda(coords)
        is_gaussian = self.num_splashes > 0
        gmm = gmm[:, is_gaussian]
    
        return feats, gmm
    











    

    def hash_index(self, coords, resolution, codebook_size):
        prime = [1, 2654435761, 805459861]
        if pow(resolution, 3) <= codebook_size:
            index = (coords[..., 0] + coords[..., 1] * resolution + coords[..., 2] * pow(resolution, 2))
        else:
            index = ((coords[..., 0] * prime[0]) ^ (coords[..., 1] * prime[1]) ^ (coords[..., 2] * prime[2])) % codebook_size
        return index

    
    def get_corners(self, coords, codebook_size, resolution):
        num_coords, coord_dim = coords.shape
        x = torch.clamp(resolution * coords, 0.0, float(resolution-1-1e-3))
        pos = torch.floor(x).long()
        x_ = x - pos
        _x = 1.0 - x_

        coeffs = torch.empty([num_coords, 8], device=coords.device)
        coeffs[:, 0] = _x[:, 0] * _x[:, 1] * _x[:, 2]
        coeffs[:, 1] = _x[:, 0] * _x[:, 1] * x_[:, 2]
        coeffs[:, 2] = _x[:, 0] * x_[:, 1] * _x[:, 2]
        coeffs[:, 3] = _x[:, 0] * x_[:, 1] * x_[:, 2]
        coeffs[:, 4] = x_[:, 0] * _x[:, 1] * _x[:, 2]
        coeffs[:, 5] = x_[:, 0] * _x[:, 1] * x_[:, 2]
        coeffs[:, 6] = x_[:, 0] * x_[:, 1] * _x[:, 2]
        coeffs[:, 7] = x_[:, 0] * x_[:, 1] * x_[:, 2]

        corners = torch.empty([num_coords, 8, coord_dim], device=coords.device).long()
        for k in range(8):
            corners[:, k, 0] = pos[:, 0] + ((k & 4) >> 2)
            corners[:, k, 1] = pos[:, 1] + ((k & 2) >> 1)
            corners[:, k, 2] = pos[:, 2] + ((k & 1) >> 0)
        
        corner_idx = self.hash_index(corners, resolution, codebook_size)
        return corners, corner_idx, coeffs


    def interpolate(self, coords):
        num_coords, coord_dim = coords.shape
        feature_dim = self.feats.shape[-1]
        mean_stds = torch.cat([self.means, self.stds], dim=-1)

        feats = torch.zeros([num_coords, feature_dim*self.num_lods], device=coords.device)
        gmms = torch.zeros([num_coords, int((self.num_splashes>0).sum())], device=coords.device)
        j = 0
        for i in range(self.num_lods):
            resolution = int(self.resolutions[i])
            # codebook_size_level = int(self.num_idxes[i])
            num_splash = int(self.num_splashes[i])

            feats_level = self.feats[self.feat_begin_idxes[i]:self.feat_begin_idxes[i+1]]
            feats_level = feats_level.view(-1, max(num_splash, 1), feature_dim)
            if num_splash:
                mean_stds_level = mean_stds[self.gau_begin_idxes[i]:self.gau_begin_idxes[i+1]]
                mean_stds_level = mean_stds_level.view(-1, num_splash, coord_dim+1)
            else:
                mean_stds_level = None

            _, corner_idx, coeffs = self.get_corners(coords, self.hashmap_size, resolution)
            corner_idx = corner_idx.view(-1)                                                    # [num_coords*8]

            if num_splash:
                coeffs = coeffs.view(num_coords, 8, 1, 1)                                       # [num_coords, 8, 1, 1]

                mean_std = torch.index_select(mean_stds_level, dim=0, index=corner_idx)         # [num_coords*8, num_splash, 4]
                mean_std = mean_std.view(num_coords, 8, num_splash, coord_dim+1)                # [num_coords, 8, num_splash, 4]
                mean = mean_std[..., :coord_dim]                                                # [num_coords, 8, num_splash, 3]
                std = mean_std[..., coord_dim:]                                                 # [num_coords, 8, num_splash, 1]
                std = torch.abs(std) 

                coords_mod = coords.view(num_coords, 1, 1, coord_dim)
                diff = coords_mod - mean
                sq_dist = torch.div(torch.pow(diff, 2), 2*torch.pow(std, 2) + 1e-7)             # [num_coords, 8, num_splash, 3]
                sq_dist = torch.sum(sq_dist, dim=-1, keepdim=True)                              # [num_coords, 8, num_splash, 1]
                gau_weights = torch.exp(-1 * sq_dist)                                           # [num_coords, 8, num_splash, 1]
                gau_norm = math.sqrt(2 * math.pi) * std
                gau_weights = torch.div(gau_weights, gau_norm + 1e-7)
                
                norm = 2.0
                dist_weights = torch.pow(torch.abs(diff/std), norm)
                dist_weights = torch.sum(dist_weights, dim=-1, keepdim=True)
                gmm, _ = torch.min((0.5*dist_weights - torch.log(coeffs+1e-7)).view(num_coords, 8*num_splash, 1), dim=-2)
                gmms[:, j] = gmm.squeeze()                                                       # [num_coords, 8, num_splash, 1]
                j += 1

                feat = torch.index_select(feats_level, dim=0, index=corner_idx)                 # [num_coords*8, num_splash+1, feature_dim]
                feat = feat.view(num_coords, 8, num_splash, feature_dim)                      # [num_coords, 8, num_splash+1, feature_dim]

                feat_comp = feat * gau_weights * coeffs                                         # [num_coords, 8, num_splash+1, feature_dim]
                feat_comp = torch.sum(feat_comp, dim=[1, 2])

            else:
                coeffs = coeffs.view(num_coords, 8, 1)

                feat = torch.index_select(feats_level, dim=0, index=corner_idx)                 # [num_coords*8, num_splash+1, feature_dim]
                feat = feat.view(num_coords, 8, feature_dim)                                    # [num_coords, 8, num_splash+1, feature_dim]
                
                feat_comp = feat * coeffs                                                       # [num_coords, 8, num_splash+1, feature_dim]
                feat_comp = torch.sum(feat_comp, dim=[1])

            # if i < lod_idx:
            feats[:, feature_dim*i:feature_dim*(i+1)] = feat_comp

        return feats, gmms