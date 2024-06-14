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
__device__ void DeviceRgbToGrayscale(const DTYPE* __restrict__ img,
                                     DTYPE* __restrict__ gray,
                                     const tensor_view_4d_t img_tv,
                                     const tensor_view_4d_t gray_tv,
                                     const ulong N)
{
    size_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    if(gid >= N)
        return;

    int n, c, h, w;
    getNCHW(n, c, h, w, gid, gray_tv.size);

    FLOAT_ACCUM r = CVT_FLOAT2ACCUM(get4DValueAt(img, img_tv, n, 0, h, w));
    FLOAT_ACCUM g = CVT_FLOAT2ACCUM(get4DValueAt(img, img_tv, n, 1, h, w));
    FLOAT_ACCUM b = CVT_FLOAT2ACCUM(get4DValueAt(img, img_tv, n, 2, h, w));

    DTYPE value = CVT_ACCUM2FLOAT(0.2989f * r + 0.587f * g + 0.114f * b);

    gray[gray_tv.offset + gid] = value;
}

template <typename DTYPE>
__device__ void DeviceRgbToGrayscaleContiguous(const DTYPE* __restrict__ img,
                                               DTYPE* __restrict__ gray,
                                               const ulong img_off,
                                               const ulong gray_off,
                                               const ulong c_stride,
                                               const ulong N)
{
    size_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    if(gid >= N)
        return;

    const ulong n   = gid / c_stride;
    const ulong idx = gid % c_stride;

    size_t img_idx = idx + img_off + n * c_stride * 3;
    FLOAT_ACCUM r  = CVT_FLOAT2ACCUM(img[img_idx]);
    FLOAT_ACCUM g  = CVT_FLOAT2ACCUM(img[img_idx + c_stride]);
    FLOAT_ACCUM b  = CVT_FLOAT2ACCUM(img[img_idx + c_stride * 2]);

    DTYPE value = CVT_ACCUM2FLOAT(0.2989f * r + 0.587f * g + 0.114f * b);

    gray[gray_off + gid] = value;
}

template <typename DTYPE>
__device__ void DeviceBlend(const DTYPE* __restrict__ img1,
                            const DTYPE* __restrict__ img2,
                            DTYPE* __restrict__ output,
                            const tensor_view_4d_t img1_tv,
                            const ulong img2_off,
                            const tensor_view_4d_t output_tv,
                            const ulong n_stride,
                            const ulong c_stride,
                            const ulong N,
                            float ratio,
                            float bound)
{
    size_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    if(gid >= N)
        return;

    const ulong n       = gid / n_stride;
    const ulong img2_id = n * c_stride + gid % c_stride;

    FLOAT_ACCUM img1_v = CVT_FLOAT2ACCUM(get4DValueAt(img1, img1_tv, gid));
    FLOAT_ACCUM img2_v = CVT_FLOAT2ACCUM(img2[img2_off + img2_id]);

    DTYPE result = CVT_ACCUM2FLOAT(clamp((ratio * img1_v + (1.0f - ratio) * img2_v), 0.0f, bound));

    set4DValueAt(output, output_tv, gid, result);
}

template <typename DTYPE>
__device__ void DeviceBlendContiguous(const DTYPE* __restrict__ img1,
                                      const DTYPE* __restrict__ img2,
                                      DTYPE* __restrict__ output,
                                      const ulong img1_off,
                                      const ulong img2_off,
                                      const ulong output_off,
                                      const ulong n_stride,
                                      const ulong c_stride,
                                      const ulong N,
                                      float ratio,
                                      float bound)
{
    size_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    if(gid >= N)
        return;

    const ulong n       = gid / n_stride;
    const ulong img2_id = n * c_stride + gid % c_stride;

    FLOAT_ACCUM img1_v = CVT_FLOAT2ACCUM(img1[img1_off + gid]);
    FLOAT_ACCUM img2_v = CVT_FLOAT2ACCUM(img2[img2_off + img2_id]);

    DTYPE value = CVT_ACCUM2FLOAT(clamp((ratio * img1_v + (1.0f - ratio) * img2_v), 0.0f, bound));

    output[output_off + gid] = value;
}

extern "C" __global__ void RGBToGrayscale(const FLOAT* __restrict__ img,
                                          FLOAT* __restrict__ gray,
                                          const tensor_view_4d_t img_tv,
                                          const tensor_view_4d_t gray_tv,
                                          const ulong N)
{
    DeviceRgbToGrayscale(img, gray, img_tv, gray_tv, N);
}

extern "C" __global__ void RGBToGrayscaleContiguous(const FLOAT* __restrict__ img,
                                                    FLOAT* __restrict__ gray,
                                                    const ulong img_off,
                                                    const ulong gray_off,
                                                    const ulong c_stride,
                                                    const ulong N)
{
    DeviceRgbToGrayscaleContiguous(img, gray, img_off, gray_off, c_stride, N);
}

extern "C" __global__ void Blend(const FLOAT* __restrict__ img1,
                                 const FLOAT* __restrict__ img2,
                                 FLOAT* __restrict__ output,
                                 const tensor_view_4d_t img1_tv,
                                 const tensor_view_4d_t img2_tv,
                                 const tensor_view_4d_t output_tv,
                                 const ulong n_stride,
                                 const ulong c_stride,
                                 const ulong N,
                                 float ratio,
                                 float bound)
{
    DeviceBlend(img1,
                img2,
                output,
                img1_tv,
                img2_tv.offset,
                output_tv,
                n_stride,
                c_stride,
                N,
                ratio,
                bound);
}

extern "C" __global__ void BlendContiguous(const FLOAT* __restrict__ img1,
                                           const FLOAT* __restrict__ img2,
                                           FLOAT* __restrict__ output,
                                           const ulong img1_off,
                                           const ulong img2_off,
                                           const ulong output_off,
                                           const ulong n_stride,
                                           const ulong c_stride,
                                           const ulong N,
                                           float ratio,
                                           float bound)
{
    DeviceBlendContiguous(
        img1, img2, output, img1_off, img2_off, output_off, n_stride, c_stride, N, ratio, bound);
}