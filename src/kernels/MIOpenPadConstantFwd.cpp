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
#include "tensor_view_5d.hpp"

template <typename T = float>
__device__ T
get5DValueAt(const T* x, const uint64_t x_dims[5], size_t n, size_t c, size_t d, size_t h, size_t w)
{
    return x[n * x_dims[1] * x_dims[2] * x_dims[3] * x_dims[4] +
             c * x_dims[2] * x_dims[3] * x_dims[4] + d * x_dims[3] * x_dims[4] + h * x_dims[4] + w];
}

extern "C" __global__ void PadConstantFwdContiguous(const INPUT_TYPE* __restrict__ x,
                                                    OUTPUT_TYPE* __restrict__ y,
                                                    const tensor_view_5d_t x_tv,
                                                    const tensor_view_5d_t y_tv,
                                                    const padding_5d_t padding,
                                                    const size_t output_size,
                                                    float value)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= output_size)
        return;

    size_t o[5];
    GET_NCDHW(o[0], o[1], o[2], o[3], o[4], gid, y_tv.size);

    bool flag = true;

    for(int i = 0; i < 5; i++)
    {
        o[i] = o[i] - padding.val[2 * i];
        flag *= o[i] < x_tv.size[i];
    }

    OUTPUT_TYPE padding_value;

#if PADDING_IS_BFP16 == 1
    padding_value = float_to_bfloat16(value);
#else
    padding_value = (OUTPUT_TYPE)value;
#endif

    y[gid] = flag ? get5DValueAt(x, x_tv.size, o[0], o[1], o[2], o[3], o[4]) : padding_value;
}
