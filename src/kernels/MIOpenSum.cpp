/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"

#if MIOPEN_USE_BFP16 == 1
#define CVT_FLOAT2ACCUM(x) (bfloat16_to_float(x))
#define CVT_ACCUM2FLOAT(x) (float_to_bfloat16(x))
#define CVT_INTEGRAL2ACCUM(x) ((_FLOAT_ACCUM)(x))
#define CVT_FP32_2FLOAT(x) (CVT_ACCUM2FLOAT(x))
#define CVT_FP32_2ACCUM(x) (x)
#endif

extern "C" __global__ void SumParallelFwdContiguous(const FLOAT* __restrict__ x,
                                                    FLOAT* __restrict__ y,
                                                    uint64_t output_numel,
                                                    uint64_t reduce_size,
                                                    uint64_t parallelism_size,
                                                    uint64_t inner_size,
                                                    bool nanPropagation)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= parallelism_size * output_numel)
        return;

    uint64_t n = inner_size * parallelism_size;

    uint64_t slice_id       = gid / n;
    uint64_t slice_local_id = gid % n;

    uint64_t input_idx = slice_id * inner_size * reduce_size + slice_local_id;

    uint64_t parallel_id = slice_local_id / inner_size;

    FLOAT_ACCUM sum = static_cast<FLOAT_ACCUM>(0);
    for(uint64_t k = parallel_id; k < reduce_size; k += parallelism_size)
    {
        FLOAT_ACCUM val = CVT_FLOAT2ACCUM(x[input_idx]);
        if(nanPropagation && isnan(val))
        {
            val = static_cast<FLOAT_ACCUM>(0);
        }
        sum += val;
        input_idx += inner_size * parallelism_size;
    }

    y[gid] = CVT_ACCUM2FLOAT(sum);
}

extern "C" __global__ void SumFwdContiguous(const FLOAT* __restrict__ x,
                                            FLOAT* __restrict__ y,
                                            uint64_t output_numel,
                                            uint64_t reduce_size,
                                            uint64_t inner_size,
                                            bool nanPropagation)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= output_numel)
        return;

    uint64_t input_idx = (gid / inner_size) * inner_size * reduce_size + gid % inner_size;

    FLOAT_ACCUM sum = static_cast<FLOAT_ACCUM>(0);
    for(uint64_t k = 0; k < reduce_size; ++k)
    {
        FLOAT_ACCUM val = CVT_FLOAT2ACCUM(x[input_idx]);
        if(nanPropagation && isnan(val))
        {
            val = static_cast<FLOAT_ACCUM>(0);
        }
        sum += val;
        input_idx += inner_size;
    }

    y[gid] = CVT_ACCUM2FLOAT(sum);
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
    // size_t gid_0    = get_global_id(0);
    // size_t gid_1    = get_group_id(1) * (get_local_size(1) * 2) + get_local_id(1);
    // size_t lid      = get_local_id(1);

    size_t gid_0 = threadIdx.x + blockIdx.x * blockDim.x;
    size_t gid_1 = blockIdx.y * blockDim.y * 2 + threadIdx.y;
    size_t lid   = threadIdx.y;

    long input_base = (gid_0 / st) * st * inner_size + gid_0 % st;

    if(gid_1 < inner_size)
    {
        long input_idx = input_base + gid_1 * st;
        tmp[lid]       = input ? input[input_off + input_idx]
                               : CVT_FLOAT2ACCUM(input_half[input_half_off + input_idx]);
    }
    else
    {
        tmp[lid] = 0;
    }
    long reduce_size = get_local_size(1);
    FLOAT val;
    if(gid_1 + reduce_size < inner_size)
    {
        long input_idx = input_base + (gid_1 + reduce_size) * st;
        val            = input ? input[input_off + input_idx]
                               : CVT_FLOAT2ACCUM(input_half[input_half_off + input_idx]);
    }
    else
    {
        val = 0;
    }
    tmp[lid] += val;
    __syncthreads();

    LocalReduceSumOpt256(tmp, lid);

    if(lid == 0)
    {
        DTYPE res = tmp[0];
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