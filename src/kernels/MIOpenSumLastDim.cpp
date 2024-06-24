/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#endif

#include "float_types.h"

#ifndef REDUCE_SIZE
#define REDUCE_SIZE 256
#endif

__device__ FLOAT_ACCUM warp_reduce_sum(FLOAT_ACCUM val)
{
    if(warpSize >= 64)
        val += __shfl_down(val, 32);
    if(warpSize >= 32)
        val += __shfl_down(val, 16);
    if(warpSize >= 16)
        val += __shfl_down(val, 8);
    if(warpSize >= 8)
        val += __shfl_down(val, 4);
    if(warpSize >= 4)
        val += __shfl_down(val, 2);
    if(warpSize >= 2)
        val += __shfl_down(val, 1);
    return val;
}

__device__ FLOAT_ACCUM block_reduce_sum(FLOAT_ACCUM val)
{
    static __shared__ FLOAT_ACCUM shared[REDUCE_SIZE / warpSize];
    auto lane = threadIdx.x % warpSize;
    auto wid  = threadIdx.x / warpSize;

    val = warp_reduce_sum(val);

    if(lane == 0)
        shared[wid] = val;
    __syncthreads();

    val = threadIdx.x < REDUCE_SIZE / warpSize ? shared[lane] : 0;
    if(wid == 0)
        val = warp_reduce_sum(val);

    return val;
}

extern "C" __global__ void SumLastDimForwardContiguous(const FLOAT* __restrict__ input,
                                                       half* __restrict__ input_half,
                                                       FLOAT* __restrict__ output,
                                                       FLOAT* __restrict__ tmp,
                                                       long inner_size,
                                                       long st,
                                                       char is_last,
                                                       long input_off,
                                                       long input_half_off,
                                                       long output_off,
                                                       char ignore_nan)
{
    size_t gid_0 = threadIdx.x + blockIdx.x * blockDim.x;
    size_t gid_1 = blockIdx.y * blockDim.y * 2 + threadIdx.y;
    size_t lid   = threadIdx.y;

    long input_base = (gid_0 / st) * st * inner_size + gid_0 % st;

    if(gid_1 < inner_size)
    {
        long input_idx = input_base + gid_1 * st;
        tmp[lid] = input ? input[input_off + input_idx] : input_half[input_half_off + input_idx];
    }
    else
    {
        tmp[lid] = 0;
    }
    long reduce_size = blockDim.y;
    FLOAT val;

    if(gid_1 + reduce_size < inner_size)
    {
        long input_idx = input_base + (gid_1 + reduce_size) * st;
        val = input ? input[input_off + input_idx] : input_half[input_half_off + input_idx];
    }
    else
    {
        val = 0;
    }
    tmp[lid] += val;
    __syncthreads();

    block_reduce_sum(tmp[lid]);

    if(lid == 0)
    {
        FLOAT res = tmp[0];
        if(is_last != 0)
        {
            output[output_off + gid_0] = res;
        }
        else
        {
            reduce_size *= 2;
            long workspace_inner_size       = (inner_size + reduce_size - 1) / reduce_size;
            long output_idx                 = gid_0 * workspace_inner_size + gid_1 / reduce_size;
            output[output_off + output_idx] = res;
        }
    }
}
