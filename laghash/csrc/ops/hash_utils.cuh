#ifndef _HASH_UTILS_CUH_
#define _HASH_UTILS_CUH_

typedef unsigned int uint;
namespace laghash
{

    static __inline__ __device__ int32_t 
    hash_index_3d(
        const int3 pos,
        const int32_t resolution,
        const int32_t codebook_size
    )
    {
        int32_t index = 0;
        constexpr uint32_t primes[3] = { 1u, 2654435761u, 805459861u };

        if (pow(resolution, 3) <= codebook_size)
            index = pos.x + pos.y * resolution + pos.z * resolution * resolution;
        else
            index = ((pos.x * primes[0]) ^ (pos.y * primes[1]) ^ (pos.z * primes[2])) % codebook_size;
        return index;
    }


    // static __inline__ __device__ int32_t 
    // hash_index_3d_alt(
    //     const int3 pos,
    //     const int32_t resolution,
    //     const int32_t codebook_size
    // ){
    //     int32_t index = 0;
    //     constexpr uint32_t primes[3] = { 3674653429u, 2097192037u, 1434869437u };

    //     if (resolution < codebook_size && resolution * resolution < codebook_size && resolution * resolution * resolution < codebook_size)
    //         index = pos.x + pos.y * resolution + pos.z * resolution * resolution;
    //     else
    //         index = ((pos.x * primes[0]) ^ (pos.y * primes[1]) ^ (pos.z * primes[2])) % codebook_size;
    //     return index;
    // }


    static __inline__ __device__ int32_t 
    hash_index_2d(
        const int2 pos,
        const int32_t resolution_x,
        const int32_t resolution_y,
        const int32_t codebook_size
    )
    {
        int32_t index = 0;
        constexpr uint32_t primes[3] = { 1u, 2654435761u };

        if (resolution_x < codebook_size && resolution_x * resolution_y < codebook_size)
            index = pos.x + pos.y * resolution_x;
        else
            index = ((pos.x * primes[0]) ^ (pos.y * primes[1])) % codebook_size;
        return index;
    }


    // static __inline__ __device__ int32_t 
    // hash_index_2d_alt(
    //     const int2 pos,
    //     const int32_t resolution_x,
    //     const int32_t resolution_y,
    //     const int32_t codebook_size
    // )
    // {
    //     int32_t index = 0;
    //     constexpr uint32_t primes[3] = { 3674653429u, 2097192037u };

    //     if (resolution_x < codebook_size && resolution_x * resolution_y < codebook_size)
    //         index = pos.x + pos.y * resolution_x;
    //     else
    //         index = ((pos.x * primes[0]) ^ (pos.y * primes[1])) % codebook_size;
    //     return index;
    // }


    static __inline__ __device__ float 
    clamp(float x, float a, float b)
    {
        return max(a, min(b, x));
    }

}
#endif