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
#include "miopen/tensor.hpp"
#include <cstddef>
#include <cstdint>
#include <sys/types.h>
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_runtime.h>
#endif

#include "miopen_cstdint.hpp"
#include "float_types.h"

// __kernel void PadConstantFwdContiguous(
//     __global DTYPE* input, __global DTYPE* output, const padding_5d padding,
//     PDTYPE value, int output_dim_size, unsigned long output_size,
//     const tensor_view_5d_t input_tv, const tensor_view_5d_t output_tv) {
//   size_t gid = get_global_id(0);
//   if (gid >= output_size) return;

//   size_t o[5];
//   GET_NCDHW(o[0], o[1], o[2], o[3], o[4], gid, output);
//   bool flag = true;

//   for (int i =0; i <5; i++){
//     o[i] = o[i] - padding.val[2*i];
//     flag *= (o[i] >= 0 && o[i] < input_tv.size[i]);
//   }

//   DTYPE val = flag ? GET_5D_VAL_AT(input, o[0], o[1], o[2], o[3], o[4]) : value;

//   output[gid + output_tv.offset] = val;
// }

// #define GET_NCDHW(n, c, d, h, w, idx, tv) \
//   {                                       \
//     ulong ncdh = (idx) / tv##_tv.size[4]; \
//     w = (idx) % tv##_tv.size[4];          \
//     ulong ncd = ncdh / tv##_tv.size[3];   \
//     h = ncdh % tv##_tv.size[3];           \
//     ulong nc = ncd / tv##_tv.size[2];     \
//     d = ncd % tv##_tv.size[2];            \
//     n = nc / tv##_tv.size[1];             \
//     c = nc % tv##_tv.size[1];             \
//   }

extern "C" __global__ void PadConstantFwdContiguous(const FLOAT_ACCUM* __restrict__ x,
                                                    FLOAT_ACCUM* __restrict__ y,
                                                    const size_t* __restrict y_dims, // needed to calculate the abosolue position in output
                                                    const size_t* __restrict__ padding,
                                                    const size_t output_size,
                                                    float value)
{
    const int gid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(gid >= output_size)
        return;
    bool flag = true;
    int o[5];

    ulong ncdh = gid / y_dims[4];
    o[4] = gid % y_dims[4];
    ulong ncd = ncdh / y_dims[3];
    o[3] = ncdh % y_dims[3];
    ulong nc = ncd / y_dims[2];
    o[2] = ncd % y_dims[2];
    o[1] = nc / y_dims[1];
    o[0] = nc % y_dims[1];
    
    for(int i = 0; i < 5; i++)
    {
        o[i] = o[i] - padding[2*i];
        flag *= (o[i] >= 0 && o[i] < y_dims[i]);
    }


    y[gid] = flag ? x[o[0] * o[1] * o[2] * o[3] * o[4] + o[1] * o[2] * o[3] * o[4] +
                      o[2] * o[3] * o[4] + o[3] * o[4] + o[4]]
                  : value;
}
