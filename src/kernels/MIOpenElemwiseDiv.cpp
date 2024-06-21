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

#include <cstddef>
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"
#include "tensor_view.hpp"

template <typename DTYPE>
__device__ void DeviceElemwiseDiv(const tensor_view_5d_t input_a_tv,
                                  const tensor_view_5d_t input_b_tv,
                                  const tensor_view_5d_t output_tv,
                                  const DTYPE* __restrict__ input_a,
                                  const DTYPE* __restrict__ input_b,
                                  DTYPE* __restrict__ output,
                                  const size_t N)
{
    size_t gid = blockDim.x * blockIdx.x + threadIdx.x;

    if(gid >= N)
        return;

    FLOAT_ACCUM a = CVT_FLOAT2ACCUM(get5DValueAt(input_a, input_a_tv, gid));
    FLOAT_ACCUM b = CVT_FLOAT2ACCUM(get5DValueAt(input_b, input_b_tv, gid));

    DTYPE value = CVT_ACCUM2FLOAT(a / b);
    set5DValueAt(output, output_tv, gid, value);
}

template <typename DTYPE>
__device__ void DeviceElemwiseDivContiguous(const size_t input_a_offset,
                                            const size_t input_b_offset,
                                            const size_t output_offset,
                                            const DTYPE* __restrict__ input_a,
                                            const DTYPE* __restrict__ input_b,
                                            DTYPE* __restrict__ output,
                                            const size_t N)
{
    size_t gid = blockDim.x * blockIdx.x + threadIdx.x;

    if(gid >= N)
        return;

    FLOAT_ACCUM a = CVT_FLOAT2ACCUM(input_a[input_a_offset + gid]);
    FLOAT_ACCUM b = CVT_FLOAT2ACCUM(input_b[input_b_offset + gid]);

    output[output_offset + gid] = CVT_ACCUM2FLOAT(a / b);
}

extern "C" __global__ void ElemwiseDivForwardContiguous(const size_t input_a_offset,
                                                        const size_t input_b_offset,
                                                        const size_t output_offset,
                                                        const size_t N,
                                                        const FLOAT* __restrict__ input_a,
                                                        const FLOAT* __restrict__ input_b,
                                                        FLOAT* __restrict__ output)
{
    DeviceElemwiseDivContiguous<FLOAT>(
        input_a_offset, input_b_offset, output_offset, input_a, input_b, output, N);
}

extern "C" __global__ void ElemwiseDivForward(const tensor_view_5d_t input_a_tv,
                                              const tensor_view_5d_t input_b_tv,
                                              const tensor_view_5d_t output_tv,
                                              const size_t N,
                                              const FLOAT* __restrict__ input_a,
                                              const FLOAT* __restrict__ input_b,
                                              FLOAT* __restrict__ output)
{
    DeviceElemwiseDiv<FLOAT>(input_a_tv, input_b_tv, output_tv, input_a, input_b, output, N);
}
