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
__device__ void DeviceImageNormalizeFwdContiguous(const DTYPE* __restrict__ input,
                                                  const DTYPE* __restrict__ mean,
                                                  const DTYPE* __restrict__ std,
                                                  DTYPE* __restrict__ output,
                                                  const size_t input_off,
                                                  const size_t mean_off,
                                                  const size_t std_off,
                                                  const size_t output_off,
                                                  const long c_stride,
                                                  const long C,
                                                  const long N)
{
    size_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    if(gid >= N)
        return;

    int c = gid / c_stride % C;

    FLOAT_ACCUM pixel  = CVT_FLOAT2ACCUM(input[gid + input_off]);
    FLOAT_ACCUM mean_p = CVT_FLOAT2ACCUM(mean[c + mean_off]);
    FLOAT_ACCUM std_p  = CVT_FLOAT2ACCUM(std[c + std_off]);
    FLOAT_ACCUM result = (pixel - mean_p) / std_p;

    output[gid + output_off] = CVT_ACCUM2FLOAT(result);
}

extern "C" __global__ void ImageNormalizeContiguous(const FLOAT* __restrict__ input,
                                                    const FLOAT* __restrict__ mean,
                                                    const FLOAT* __restrict__ std,
                                                    FLOAT* __restrict__ output,
                                                    const size_t input_off,
                                                    const size_t mean_off,
                                                    const size_t std_off,
                                                    const size_t output_off,
                                                    const size_t c_stride,
                                                    const size_t C,
                                                    const size_t N)
{
    DeviceImageNormalizeFwdContiguous<FLOAT>(
        input, mean, std, output, input_off, mean_off, std_off, output_off, c_stride, C, N);
}