#include <iostream>
#include <stdio.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include "hash_utils.cuh"

#define FEATURE_DIM 2

namespace laghash
{
    typedef unsigned int uint;

    template<typename scalar_t> // , int32_t FEATURE_DIM, int32_t NUM_GAUS
    __global__ void
    laghash_interpolate_3d_cuda_kernel(
        const int64_t num_coords,
        const int32_t codebook_size,
        const int32_t feature_dim,
        const int32_t resolution,
        const int32_t lod_idx,
        const int32_t num_lods,
        const int32_t num_gaus,
        const float* __restrict__ coords,
        const scalar_t* __restrict__ codebook,
        const scalar_t* __restrict__ means,
        const scalar_t* __restrict__ stds,
        const int64_t *codebook_first_idx,
        const int64_t *gau_first_idx,
        scalar_t* __restrict__ feats,
        scalar_t* __restrict__ gmms
    )
    {
        uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
        int64_t stride = blockDim.x * gridDim.x;
        const float pi = 3.14159265358979323846;

        codebook = codebook + codebook_first_idx[0]*FEATURE_DIM;
        means = means + gau_first_idx[0]*3;
        stds = stds + gau_first_idx[0];

        for (int64_t i=tidx; i<num_coords; i+=stride) 
        {
            float3 pos3d = make_float3(coords[i*3+0], coords[i*3+1], coords[i*3+2]);
            float3 x = make_float3(clamp(resolution*pos3d.x, 0, resolution-1-1e-3), 
                                   clamp(resolution*pos3d.y, 0, resolution-1-1e-3), 
                                   clamp(resolution*pos3d.z, 0, resolution-1-1e-3));
            int3 pos = make_int3(floor(x.x), floor(x.y), floor(x.z));
            float3 x_ = make_float3(x.x - static_cast<float>(pos.x), 
                                    x.y - static_cast<float>(pos.y), 
                                    x.z - static_cast<float>(pos.z));
            float3 _x = make_float3(1.0 - x_.x, 1.0 - x_.y, 1.0 - x_.z);

            float coeffs[8];
            coeffs[0] = _x.x * _x.y * _x.z;
            coeffs[1] = _x.x * _x.y * x_.z;
            coeffs[2] = _x.x * x_.y * _x.z;
            coeffs[3] = _x.x * x_.y * x_.z;
            coeffs[4] = x_.x * _x.y * _x.z;
            coeffs[5] = x_.x * _x.y * x_.z;
            coeffs[6] = x_.x * x_.y * _x.z;
            coeffs[7] = x_.x * x_.y * x_.z;

            int32_t corner_idx[8];
            #pragma unroll
            for (int k=0; k<8; ++k) 
            {
                int3 corner;
                corner.x = pos.x + ((k & 4) >> 2);
                corner.y = pos.y + ((k & 2) >> 1);
                corner.z = pos.z + ((k & 1) >> 0);
                corner_idx[k] = hash_index_3d(corner, resolution, codebook_size);
            }

            float feat[FEATURE_DIM] = {0.0};  // FEATURE_DIM = 2
            float gmm = 1e5;

            #pragma unroll
            for (int k=0; k<8; ++k)
            {
                for (int l=0; l<num_gaus; ++l)
                {
                    float mean_x = static_cast<float>(means[corner_idx[k]*num_gaus*3 + l*3 + 0]);
                    float mean_y = static_cast<float>(means[corner_idx[k]*num_gaus*3 + l*3 + 1]);
                    float mean_z = static_cast<float>(means[corner_idx[k]*num_gaus*3 + l*3 + 2]);
                    float std = fabsf(static_cast<float>(stds[corner_idx[k]*num_gaus + l]));

                    float distx = pos3d.x - mean_x;
                    float disty = pos3d.y - mean_y;
                    float distz = pos3d.z - mean_z;
                    float dist_sq = powf(distx, 2) +  powf(disty, 2) +  powf(distz, 2);

                    float weight = expf(-1 * dist_sq / (2 * powf(std, 2) + 1e-7));
                    weight /=  sqrtf(2 * pi * powf(std, 2) + 1e-7);
                    gmm = fminf(gmm, -logf(coeffs[k]+1e-5) + 0.5*dist_sq/powf(std, 2));

                    #pragma unroll
                    for (uint64_t j=0; j<FEATURE_DIM; ++j)
                    {
                        float ft = static_cast<float>(codebook[corner_idx[k]*num_gaus*FEATURE_DIM + l*FEATURE_DIM + j]);
                        feat[j] += coeffs[k]*weight*ft;
                    }
                }
            }

            gmms[num_lods*i + lod_idx] = static_cast<scalar_t>(gmm);
            #pragma unroll
            for (uint64_t j=0; j<FEATURE_DIM; ++j)
            {
                feats[num_lods*i*FEATURE_DIM + lod_idx*FEATURE_DIM + j] = static_cast<scalar_t>(feat[j]);
            }
        }
    }


    template<typename scalar_t> // , int32_t FEATURE_DIM, int32_t NUM_GAUS
    __global__ void
    laghash_interpolate_3d_backward_cuda_kernel(
        const int64_t num_coords,
        const int32_t codebook_size,
        const int32_t feature_dim,
        const int32_t resolution,
        const int32_t lod_idx,
        const int32_t num_lods,
        const int32_t num_gaus,
        const bool require_grad_std,
        const float* __restrict__ coords,
        const scalar_t* __restrict__ codebook,
        const scalar_t* __restrict__ means,
        const scalar_t* __restrict__ stds,
        const int64_t *__restrict__ codebook_first_idx,
        const int64_t *__restrict__ gau_first_idx,
        const scalar_t* __restrict__ grad_feats,            // N, FEATURE_DIM*num_lods
        const scalar_t* __restrict__ grad_gmms,             // N, num_lods
        scalar_t* __restrict__ grad_codebook,               // codebook_size, FEATURE_DIM
        scalar_t* __restrict__ grad_means,                  // codebook_size, 3
        scalar_t* __restrict__ grad_stds                    // codebook_size, 1
    )
    {
        uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
        int64_t stride = blockDim.x * gridDim.x;
        const float pi = 3.14159265358979323846;

        codebook = codebook + codebook_first_idx[0]*FEATURE_DIM;
        means = means + gau_first_idx[0]*3;
        stds = stds + gau_first_idx[0];
        
        grad_codebook = grad_codebook + codebook_first_idx[0]*FEATURE_DIM;
        grad_means = grad_means + gau_first_idx[0]*3;
        grad_stds = grad_stds + gau_first_idx[0];

        for (int64_t i=tidx; i<num_coords; i+=stride) 
        {
            float3 pos3d = make_float3(coords[i*3+0], coords[i*3+1], coords[i*3+2]);
            float3 x = make_float3(clamp(resolution*pos3d.x, 0, resolution-1-1e-3), 
                                   clamp(resolution*pos3d.y, 0, resolution-1-1e-3), 
                                   clamp(resolution*pos3d.z, 0, resolution-1-1e-3));
            int3 pos = make_int3(floor(x.x), floor(x.y), floor(x.z));
            float3 x_ = make_float3(x.x - static_cast<float>(pos.x), 
                                    x.y - static_cast<float>(pos.y), 
                                    x.z - static_cast<float>(pos.z));
            float3 _x = make_float3(1.0 - x_.x, 1.0 - x_.y, 1.0 - x_.z);

            float coeffs[8];
            coeffs[0] = _x.x * _x.y * _x.z;
            coeffs[1] = _x.x * _x.y * x_.z;
            coeffs[2] = _x.x * x_.y * _x.z;
            coeffs[3] = _x.x * x_.y * x_.z;
            coeffs[4] = x_.x * _x.y * _x.z;
            coeffs[5] = x_.x * _x.y * x_.z;
            coeffs[6] = x_.x * x_.y * _x.z;
            coeffs[7] = x_.x * x_.y * x_.z;

            int32_t corner_idx[8];
            #pragma unroll
            for (int k=0; k<8; ++k) 
            {
                int3 corner;
                corner.x = pos.x + ((k & 4) >> 2);
                corner.y = pos.y + ((k & 2) >> 1);
                corner.z = pos.z + ((k & 1) >> 0);
                corner_idx[k] = hash_index_3d(corner, resolution, codebook_size);
            }
            float pd_gmm_meanx = 0, pd_gmm_meany = 0, pd_gmm_meanz = 0, pd_gmm_std = 0;
            int ck = -1, cl = -1;
            float gmm = 1e5;

            #pragma unroll
            for (int k=0; k<8; ++k)
            {
                for (int l=0; l<num_gaus; ++l)
                {
                    float mean_x = static_cast<float>(means[corner_idx[k]*num_gaus*3 + l*3 + 0]);
                    float mean_y = static_cast<float>(means[corner_idx[k]*num_gaus*3 + l*3 + 1]);
                    float mean_z = static_cast<float>(means[corner_idx[k]*num_gaus*3 + l*3 + 2]);
                    float std = fabsf(static_cast<float>(stds[corner_idx[k]*num_gaus + l]));

                    float distx = pos3d.x - mean_x;
                    float disty = pos3d.y - mean_y;
                    float distz = pos3d.z - mean_z;
                    float dist_sq = powf(distx, 2) +  powf(disty, 2) +  powf(distz, 2);

                    float pd_meanx = distx / (powf(std, 2) + 1e-7);
                    float pd_meany = disty / (powf(std, 2) + 1e-7);
                    float pd_meanz = distz / (powf(std, 2) + 1e-7);
                    float pd_std = dist_sq / (powf(std, 3) + 1e-7) - 1.0 / std;

                    float weight = expf(-1 * dist_sq / (2 * powf(std, 2) + 1e-7));
                    weight /=  sqrtf(2 * pi) * std + 1e-7;

                    float log_weight = -logf(coeffs[k]+1e-5) + 0.5*dist_sq/powf(std, 2);
                    gmm = fminf(gmm, log_weight);
                    if (gmm == log_weight)
                    {
                        ck = k;
                        cl = l;
                        pd_gmm_meanx = -distx / (powf(std, 2) + 1e-7);
                        pd_gmm_meany = -disty / (powf(std, 2) + 1e-7);
                        pd_gmm_meanz = -distz / (powf(std, 2) + 1e-7);
                    }

                    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
                    if (std::is_same<scalar_t, at::Half>::value) 
                    {
                        #pragma unroll
                        for (uint32_t j=0; j<FEATURE_DIM; j += 2) 
                        {
                            uint64_t feat_idx = i*num_lods*FEATURE_DIM + lod_idx*FEATURE_DIM + j;
                            __half2 grad = reinterpret_cast<const __half2*>(grad_feats)[feat_idx/2];
                            __half2 grad_feat = __floats2half2_rn(__half2float(grad.x)*coeffs[k]*weight,
                                                                  __half2float(grad.y)*coeffs[k]*weight);
                            atomicAdd((__half2*)(grad_codebook + (corner_idx[k]*num_gaus*FEATURE_DIM + l*FEATURE_DIM + j)), grad_feat);

                            __half2 feat = reinterpret_cast<const __half2*>(codebook)[(corner_idx[k]*num_gaus*FEATURE_DIM + l*FEATURE_DIM + j)/2];
                            float pd_grad = __half2float(grad_feat.x) * __half2float(feat.x) + __half2float(grad_feat.y) * __half2float(feat.y);
                            __half grad_meanx = __float2half(pd_grad*pd_meanx);
                            __half grad_meany = __float2half(pd_grad*pd_meany);
                            __half grad_meanz = __float2half(pd_grad*pd_meanz);

                            atomicAdd((__half*)(grad_means + (corner_idx[k]*num_gaus*3 + l*3 + 0)), grad_meanx);
                            atomicAdd((__half*)(grad_means + (corner_idx[k]*num_gaus*3 + l*3 + 1)), grad_meany);
                            atomicAdd((__half*)(grad_means + (corner_idx[k]*num_gaus*3 + l*3 + 2)), grad_meanz);
                        }
                    }
                    else
                    #endif
                    {
                        #pragma unroll
                        for (uint32_t j=0; j<FEATURE_DIM; ++j)
                        {
                            uint64_t feat_idx = i*num_lods*FEATURE_DIM + lod_idx*FEATURE_DIM + j;
                            float grad_feat = grad_feats[feat_idx] * coeffs[k] * weight;
                            atomicAdd((float*)(grad_codebook + (corner_idx[k]*num_gaus*FEATURE_DIM + l*FEATURE_DIM + j)), grad_feat);

                            float feat = static_cast<float>(codebook[corner_idx[k]*num_gaus*FEATURE_DIM + l*FEATURE_DIM + j]);
                            atomicAdd((float*)(grad_means + (corner_idx[k]*num_gaus*3 + l*3 + 0)), grad_feat*feat*pd_meanx);
                            atomicAdd((float*)(grad_means + (corner_idx[k]*num_gaus*3 + l*3 + 1)), grad_feat*feat*pd_meany);
                            atomicAdd((float*)(grad_means + (corner_idx[k]*num_gaus*3 + l*3 + 2)), grad_feat*feat*pd_meanz);
                            if (require_grad_std)
                                atomicAdd((float*)(grad_stds + (corner_idx[k]*num_gaus + l)), grad_feat*feat*pd_std);
                        }
                    }
                }
            }

            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
            if (std::is_same<scalar_t, at::Half>::value) 
            {
                uint64_t gmm_idx = i*num_lods + lod_idx;
                __half grad_gmm = reinterpret_cast<const __half*>(grad_gmms)[gmm_idx];
                __half grad_means_x = __float2half(__half2float(grad_gmm)*pd_gmm_meanx);
                __half grad_means_y = __float2half(__half2float(grad_gmm)*pd_gmm_meany);
                __half grad_means_z = __float2half(__half2float(grad_gmm)*pd_gmm_meanz);
                atomicAdd((__half*)(grad_means + (corner_idx[ck]*num_gaus*3 + cl*3 + 0)), grad_means_x);
                atomicAdd((__half*)(grad_means + (corner_idx[ck]*num_gaus*3 + cl*3 + 1)), grad_means_y);
                atomicAdd((__half*)(grad_means + (corner_idx[ck]*num_gaus*3 + cl*3 + 2)), grad_means_z);

                // if (require_grad_std)
                // {
                //     __half2 grad_std = __floats2half2_rn(__half2float(grad_gmm.x)*pd_gmm_std, 0);
                //     atomicAdd((__half2*)(grad_stds + (corner_idx[ck]*num_gaus + cl)), grad_std);
                // }
            }
            else
            #endif
            {
                uint64_t gmm_idx = i*num_lods + lod_idx;
                float grad_gmm = grad_gmms[gmm_idx];
                atomicAdd((float*)(grad_means + (corner_idx[ck]*num_gaus*3 + cl*3 + 0)), grad_gmm*pd_gmm_meanx);
                atomicAdd((float*)(grad_means + (corner_idx[ck]*num_gaus*3 + cl*3 + 1)), grad_gmm*pd_gmm_meany);
                atomicAdd((float*)(grad_means + (corner_idx[ck]*num_gaus*3 + cl*3 + 2)), grad_gmm*pd_gmm_meanz);
                if (require_grad_std)
                    atomicAdd((float*)(grad_stds + (corner_idx[ck]*num_gaus + cl)), grad_gmm*pd_gmm_std);
            }
        }
    }


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
        at::Tensor gmms)
    {
        int num_threads = 512;

        if (coord_dim == 3) 
        {
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(feats.type(), "laghash_interpolate_3d_cuda", ([&] {
                const at::cuda::OptionalCUDAGuard device_guard(at::device_of(feats));
                auto stream = at::cuda::getCurrentCUDAStream();
                // const int32_t FEATURE_DIM = 2;
                // const int32_t NUM_GAUS = 9; <scalar_t, FEATURE_DIM, GAUS>
                laghash_interpolate_3d_cuda_kernel<<<(num_coords + num_threads - 1) / num_threads, num_threads, 0, stream>>>(
                    num_coords,
                    codebook_size,
                    feature_dim,
                    resolution,
                    lod_idx,
                    num_lods,
                    num_gaus,
                    coords.data_ptr<float>(),
                    codebook.data_ptr<scalar_t>(),
                    means.data_ptr<scalar_t>(),
                    stds.data_ptr<scalar_t>(),
                    codebook_first_idx.data_ptr<int64_t>(),
                    gau_first_idx.data_ptr<int64_t>(),
                    feats.data_ptr<scalar_t>(),
                    gmms.data_ptr<scalar_t>()
                );
            }));
        }
    }


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
        at::Tensor grad_stds)
    {
        int num_threads = 512;

        if (coord_dim == 3) 
        {
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_feats.type(), "laghash_interpolate_3d_backward_cuda", ([&] {
                const at::cuda::OptionalCUDAGuard device_guard(at::device_of(grad_feats));
                auto stream = at::cuda::getCurrentCUDAStream();
                // const int32_t FEATURE_DIM = 2;
                // const int32_t NUM_GAUS = 9; <scalar_t, FEATURE_DIM, NUM_GAUS>
                laghash_interpolate_3d_backward_cuda_kernel<<<(num_coords + num_threads - 1) / num_threads, num_threads, 0, stream>>>(
                    num_coords,
                    codebook_size,
                    feature_dim,
                    resolution,
                    lod_idx,
                    num_lods,
                    num_gaus,
                    require_grad_std,
                    coords.data_ptr<float>(),
                    codebook.data_ptr<scalar_t>(),
                    means.data_ptr<scalar_t>(),
                    stds.data_ptr<scalar_t>(),
                    codebook_first_idx.data_ptr<int64_t>(),
                    gau_first_idx.data_ptr<int64_t>(),
                    grad_feats.data_ptr<scalar_t>(),
                    grad_gmms.data_ptr<scalar_t>(),
                    grad_codebook.data_ptr<scalar_t>(),
                    grad_means.data_ptr<scalar_t>(),
                    grad_stds.data_ptr<scalar_t>()
                );
            }));
        }
    }
} // namespace laghash