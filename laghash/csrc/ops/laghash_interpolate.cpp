#include <ATen/ATen.h>
#include <vector>
#include <iostream>

namespace laghash
{
    void hash_interpolate_cuda_impl(
        int64_t num_coords, 
        int32_t codebook_size,
        int32_t feature_dim,
        int32_t resolution,
        int32_t lod_idx,
        int32_t num_lods,
        int32_t coord_dim,
        at::Tensor coords,
        at::Tensor codebook,
        at::Tensor codebook_first_idx,
        at::Tensor feats);


    void hash_interpolate_backward_cuda_impl(
        int64_t num_coords, 
        int32_t codebook_size,
        int32_t feature_dim,
        int32_t resolution,
        int32_t lod_idx,
        int32_t num_lods,
        int32_t coord_dim,
        at::Tensor coords,
        at::Tensor codebook,
        at::Tensor codebook_first_idx,
        at::Tensor grad_feats,
        at::Tensor grad_codebook);


    void laghash_interpolate_cuda_impl(
        int64_t num_coords, 
        int32_t codebook_size,
        int32_t feature_dim,
        int32_t resolution,
        int32_t lod_idx,
        int32_t num_lods,
        int32_t coord_dim,
        int32_t num_gaus,
        at::Tensor coords,
        at::Tensor codebook,
        at::Tensor means,
        at::Tensor stds,
        at::Tensor codebook_first_idx,
        at::Tensor gau_first_idx,
        at::Tensor feats,
        at::Tensor gmms);


    void laghash_interpolate_backward_cuda_impl(
        int64_t num_coords, 
        int32_t codebook_size,
        int32_t feature_dim,
        int32_t resolution,
        int32_t lod_idx,
        int32_t num_lods,
        int32_t coord_dim,
        int32_t num_gaus,
        bool require_grad_std,
        at::Tensor coords,
        at::Tensor codebook,
        at::Tensor means,
        at::Tensor stds,
        at::Tensor codebook_first_idx,
        at::Tensor gau_first_idx,
        at::Tensor grad_feats,
        at::Tensor grad_gmms,
        at::Tensor grad_codebook,
        at::Tensor grad_means,
        at::Tensor grad_stds);


    std::vector<at::Tensor> laghash_interpolate_cuda(
        at::Tensor coords,
        at::Tensor codebook,
        at::Tensor means,
        at::Tensor stds,
        at::Tensor codebook_first_idxes,
        at::Tensor gau_first_idxes,
        at::Tensor resolutions,
        at::Tensor num_gaus,
        float codebook_bitwidth)
    {
        #ifdef WITH_CUDA
            int64_t num_coords = coords.size(0);
            int32_t coord_dim = coords.size(1);
            int32_t feature_dim = codebook.size(-1);
            int32_t num_lods = resolutions.size(0);
            int32_t codebook_size = pow(2, codebook_bitwidth);

            at::Tensor feats = at::zeros({num_coords, feature_dim * num_lods}, codebook.options()); 
            at::Tensor gmms = at::zeros({num_coords, num_lods}, codebook.options());            
            for (int32_t i=0; i < num_lods; ++i) 
            {
                int32_t res = resolutions[i].item<int>();
                int32_t num_g = num_gaus[i].item<int>();

                if (num_g == 0)
                    hash_interpolate_cuda_impl(num_coords, codebook_size, feature_dim, res, i, num_lods, 
                        coord_dim, coords, codebook, codebook_first_idxes[i], feats);
                else
                    laghash_interpolate_cuda_impl(num_coords, codebook_size, feature_dim, res, i, num_lods,
                        coord_dim, num_g, coords, codebook, means, stds, codebook_first_idxes[i], 
                        gau_first_idxes[i], feats, gmms);
            }
            return {feats, gmms};
        #else
            AT_ERROR(__func__);
        #endif  // WITH_CUDA
    }


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
        bool require_grad_std) 
    {
        #ifdef WITH_CUDA
            int64_t num_coords = coords.size(0);  
            int32_t coord_dim = coords.size(1);
            int32_t feature_dim = codebook.size(-1);
            int32_t num_lods = resolutions.size(0);
            int32_t codebook_size = pow(2, codebook_bitwidth);

            at::Tensor grad_codebook = at::zeros_like(codebook);
            at::Tensor grad_means = at::zeros_like(means);
            at::Tensor grad_stds = at::zeros_like(stds);
            
            for (int32_t i=0; i < num_lods; ++i) 
            {
                int32_t res = resolutions[i].item<int>();
                int32_t num_g = num_gaus[i].item<int>();

                if (num_g == 0)
                    hash_interpolate_backward_cuda_impl(num_coords, codebook_size, feature_dim, 
                        res, i, num_lods, coord_dim, coords, codebook, codebook_first_idxes[i], 
                        grad_feats, grad_codebook);
                else
                    laghash_interpolate_backward_cuda_impl(num_coords, codebook_size, feature_dim, 
                        res, i, num_lods, coord_dim, num_g, require_grad_std, coords, codebook, 
                        means, stds, codebook_first_idxes[i], gau_first_idxes[i], grad_feats, grad_gmms, 
                        grad_codebook, grad_means, grad_stds);
            }
            return {grad_codebook, grad_means, grad_stds};
        #else
            AT_ERROR(__func__);
        #endif  // WITH_CUDA
    }
}