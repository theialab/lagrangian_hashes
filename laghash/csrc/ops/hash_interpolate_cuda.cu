#include <iostream>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include "hash_utils.cuh"

#define FEATURE_DIM 2

namespace laghash 
{
    typedef unsigned int uint;

    template<typename scalar_t>
    __global__ void
    hash_interpolate_3d_cuda_kernel(
        const int64_t num_coords,
        const int32_t codebook_size,
        const int64_t feature_dim,
        const int32_t resolution,
        const int32_t lod_idx,
        const int32_t num_lods,
        const float* __restrict__ coords,
        const scalar_t* __restrict__ codebook,
        const int64_t *codebook_first_idx,
        scalar_t* __restrict__ feats
    )
    {
        uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
        int64_t stride = blockDim.x * gridDim.x;

        codebook = codebook + codebook_first_idx[0]*FEATURE_DIM; 

        for (int64_t i=tidx; i<num_coords; i+=stride) 
        {
            float3 x = make_float3(clamp(resolution*coords[i*3+0], 0, resolution-1-1e-3), 
                                   clamp(resolution*coords[i*3+1], 0, resolution-1-1e-3), 
                                   clamp(resolution*coords[i*3+2], 0, resolution-1-1e-3));
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
            
            int3 corners[8];
            int32_t corner_idx[8];
            #pragma unroll
            for (int k=0; k<8; ++k) 
            {
                corners[k].x = pos.x + ((k & 4) >> 2);
                corners[k].y = pos.y + ((k & 2) >> 1);
                corners[k].z = pos.z + ((k & 1) >> 0);
                corner_idx[k] = hash_index_3d(corners[k], resolution, codebook_size);
            }

            float feat[FEATURE_DIM] = {0.0};  // feature_dim = 2

            #pragma unroll
            for (int k=0; k<8; ++k)
            {
                #pragma unroll
                for (uint64_t j=0; j<FEATURE_DIM; ++j)
                {
                    float ft = static_cast<float>(codebook[corner_idx[k] * FEATURE_DIM + j]);
                    feat[j] += coeffs[k] * ft;
                }
            }

            #pragma unroll
            for (uint64_t j=0; j<FEATURE_DIM; ++j)
            {
                feats[num_lods*i*FEATURE_DIM + lod_idx*FEATURE_DIM + j] = static_cast<scalar_t>(feat[j]);
            }
        }
    } 


    template<typename scalar_t>
    __global__ void
    hash_interpolate_3d_backward_cuda_kernel(
        const int64_t num_coords,
        const int32_t codebook_size,
        const int64_t feature_dim,
        const int32_t resolution,
        const int32_t lod_idx,
        const int32_t num_lods,
        const float* __restrict__ coords,
        const scalar_t* __restrict__ codebook,
        const int64_t *__restrict__ codebook_first_idx,
        const scalar_t* __restrict__ grad_output, // N, feature_dim*num_lods
        scalar_t* __restrict__ grad_codebook // codebook_size, feature_dim
    )
    {
        uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
        int64_t stride = blockDim.x * gridDim.x;

        grad_codebook = grad_codebook + codebook_first_idx[0]*FEATURE_DIM;
        codebook = codebook + codebook_first_idx[0]*FEATURE_DIM; 

        for (int64_t i=tidx; i<num_coords; i+=stride) 
        {    
            float3 x = make_float3(clamp(resolution*coords[i*3+0], 0, resolution-1-1e-3), 
                                   clamp(resolution*coords[i*3+1], 0, resolution-1-1e-3), 
                                   clamp(resolution*coords[i*3+2], 0, resolution-1-1e-3));
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
            for (int j=0; j<8; ++j) 
            {
                int3 corner;
                corner.x = pos.x + ((j & 4) >> 2);
                corner.y = pos.y + ((j & 2) >> 1);
                corner.z = pos.z + ((j & 1) >> 0);
                corner_idx[j] = hash_index_3d(corner, resolution, codebook_size);
            }
    
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
            if (std::is_same<scalar_t, at::Half>::value) 
            {
                #pragma unroll
                for (uint64_t j=0; j<FEATURE_DIM; j += 2) 
                {
                    #pragma unroll
                    for (int k=0; k<8; ++k) 
                    {
                        uint64_t _idx = i*num_lods*FEATURE_DIM + lod_idx*FEATURE_DIM + j;
                        __half2 grad = reinterpret_cast<const __half2*>(grad_output)[_idx/2];
                        grad = __floats2half2_rn(__half2float(grad.x) * coeffs[k],
                                                 __half2float(grad.y) * coeffs[k]);
                        atomicAdd((__half2*)(grad_codebook + (corner_idx[k]*FEATURE_DIM + j)), grad);
                    }
                }
            }
            else
            #endif
            {
                #pragma unroll
                for (uint64_t j=0; j<FEATURE_DIM; ++j) 
                {
                    #pragma unroll
                    for (int k=0; k<8; ++k) {
                        float grad =
                            grad_output[i*num_lods*FEATURE_DIM + lod_idx*FEATURE_DIM + j] * coeffs[k];
                        atomicAdd((float*)(grad_codebook + (corner_idx[k]*FEATURE_DIM + j)), grad);
                    }
                }
            }
        }
    }


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
        at::Tensor feats)
    {
        int num_threads = 512;
        
        if (coord_dim == 3)
        {
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(feats.type(), "hash_interpolate_3d_cuda", ([&] {
                const at::cuda::OptionalCUDAGuard device_guard(at::device_of(feats));
                auto stream = at::cuda::getCurrentCUDAStream();
                hash_interpolate_3d_cuda_kernel<<<(num_coords + num_threads - 1) / num_threads, num_threads, 0, stream>>>(
                    num_coords,
                    codebook_size,
                    feature_dim,
                    resolution,
                    lod_idx,
                    num_lods,
                    coords.data_ptr<float>(),
                    codebook.data_ptr<scalar_t>(),
                    codebook_first_idx.data_ptr<int64_t>(),
                    feats.data_ptr<scalar_t>()
                );
            }));
        }
    }


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
        at::Tensor grad_codebook)
    {
        int num_threads = 512;

        if (coord_dim == 3) 
        {
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_feats.type(), "hash_interpolate_3d_backward_cuda", ([&] {
                const at::cuda::OptionalCUDAGuard device_guard(at::device_of(grad_feats));
                auto stream = at::cuda::getCurrentCUDAStream();
                hash_interpolate_3d_backward_cuda_kernel<<<(num_coords + num_threads - 1) / num_threads, num_threads, 0, stream>>>(
                    num_coords,
                    codebook_size,
                    feature_dim,
                    resolution,
                    lod_idx,
                    num_lods,
                    coords.data_ptr<float>(),
                    codebook.data_ptr<scalar_t>(),
                    codebook_first_idx.data_ptr<int64_t>(),
                    grad_feats.data_ptr<scalar_t>(),
                    grad_codebook.data_ptr<scalar_t>()
                );
            }));
        }
    }

} // namespace laghash