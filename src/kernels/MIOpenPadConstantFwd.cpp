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
#include <cstdio>
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#endif

#include "float_types.h"

#define GET_NCDHW(n, c, d, h, w, idx, size) \
    {                                       \
        ulong ncdh = (idx) / size[4];       \
        w          = (idx) % size[4];       \
        ulong ncd  = ncdh / size[3];        \
        h          = ncdh % size[3];        \
        ulong nc   = ncd / size[2];         \
        d          = ncd % size[2];         \
        n          = nc / size[1];          \
        c          = nc % size[1];          \
    }

template <typename T = float>
__device__ T
get5DValueAt(const T* x, const size_t* x_dims, size_t n, size_t c, size_t d, size_t h, size_t w)
{
    return x[n * x_dims[1] * x_dims[2] * x_dims[3] * x_dims[4] +
             c * x_dims[2] * x_dims[3] * x_dims[4] + d * x_dims[3] * x_dims[4] + h * x_dims[4] + w];
}

extern "C" __global__ void PadConstantFwdContiguous(
    const INPUT_TYPE* __restrict__ x,
    OUTPUT_TYPE* __restrict__ y,
    const size_t* __restrict__ x_dims,
    const size_t* __restrict__ y_dims,
    const size_t* __restrict__ padding,
    const size_t output_size,
    float value)
{
    //   size_t gid = get_global_id(0);
    //   if (gid >= output_size) return;
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    if(gid >= output_size)
        return;

    //   size_t o[5];
    size_t o[5];

    //   GET_NCDHW(o[0], o[1], o[2], o[3], o[4], gid, output);
    GET_NCDHW(o[0], o[1], o[2], o[3], o[4], gid, y_dims);

    //   bool flag = true;
    bool flag = true;

    //   for (int i =0; i <5; i++){
    //     o[i] = o[i] - padding.val[2*i];
    //     flag *= (o[i] >= 0 && o[i] < input_tv.size[i]);
    //   }
    for(int i = 0; i < 5; i++)
    {
        o[i] = o[i] - padding[2 * i];
        flag *= (o[i] >= 0 && o[i] < x_dims[i]);
    }

    //   DTYPE val = flag ? GET_5D_VAL_AT(input, o[0], o[1], o[2], o[3], o[4]) : value;
    y[gid] = flag ? get5DValueAt(x, x_dims, o[0], o[1], o[2], o[3], o[4]) : value;
}
