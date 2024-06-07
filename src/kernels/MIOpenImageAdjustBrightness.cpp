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
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"
#include "tensor_view.hpp"

template <typename DTYPE>
__device__ DTYPE clamp(DTYPE val, DTYPE min, DTYPE max)
{
    val = val < min ? min : val;
    val = val > max ? max : val;
    return val;
}

template <typename DTYPE>
__device__ void DeviceImageAdjustBrightness(DTYPE* input,
                                            DTYPE* output,
                                            tensor_view_4d_t input_tv,
                                            tensor_view_4d_t output_tv,
                                            size_t N,
                                            float brightness_factor)
{
    size_t gid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(gid >= N)
        return;

    DTYPE pixel    = get4DValueAt(input, input_tv.stride, gid);
    FLOAT_ACCUM fp = CVT_FLOAT2ACCUM(pixel);

    DTYPE result = clamp(fp * brightness_factor, 0.0f, 1.0f);
    set4DValueAt(output, output_tv, gid, CVT_ACCUM2FLOAT(result));
}

template <typename DTYPE>
__device__ void DeviceImageAdjustBrightnessContiguous(DTYPE* input,
                                                      DTYPE* output,
                                                      size_t input_off,
                                                      size_t output_off,
                                                      size_t N,
                                                      float brightness_factor)
{
    size_t gid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(gid >= N)
        return;

    DTYPE pixel    = input[input_off + gid];
    FLOAT_ACCUM fp = CVT_FLOAT2ACCUM(pixel);

    DTYPE result             = clamp(fp * brightness_factor, 0.0f, 1.0f);
    output[output_off + gid] = CVT_ACCUM2FLOAT(result);
}

extern "C" __global__ void ImageAdjustBrightness(FLOAT* input,
                                                 FLOAT* output,
                                                 tensor_view_4d_t input_tv,
                                                 tensor_view_4d_t output_tv,
                                                 size_t N,
                                                 float brightness_factor)
{
    DeviceImageAdjustBrightness<FLOAT>(input, output, input_tv, output_tv, N, brightness_factor);
}

extern "C" __global__ void ImageAdjustBrightnessContiguous(FLOAT* input,
                                                           FLOAT* output,
                                                           size_t input_off,
                                                           size_t output_off,
                                                           size_t N,
                                                           float brightness_factor)
{
    DeviceImageAdjustBrightnessContiguous<FLOAT>(
        input, output, input_off, output_off, N, brightness_factor);
}