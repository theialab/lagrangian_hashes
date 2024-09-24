#pragma once

#include <ATen/ATen.h>
#include <vector>

namespace laghash
{
    std::vector<at::Tensor> laghash_interpolate_cuda(
        at::Tensor coords,
        at::Tensor codebook,
        at::Tensor means,
        at::Tensor stds,
        at::Tensor codebook_first_idxes,
        at::Tensor gau_first_idxes,
        at::Tensor resolutions,
        at::Tensor num_gaus,
        float codebook_bitwidth);

    std::vector<at::Tensor> laghash_interpolate_backward_cuda(
        at::Tensor coords,
        at::Tensor codebook,
        at::Tensor means,
        at::Tensor stds,
        at::Tensor codebook_first_idxes,
        at::Tensor gau_first_idxes,
        at::Tensor resolutions,
        at::Tensor grad_feats,
        at::Tensor grad_gmms,
        at::Tensor num_gaus,
        float codebook_bitwidth,
        bool require_grad_std);
}