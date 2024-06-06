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
__device__ void
convertRGBToHSV(const DTYPE r, const DTYPE g, const DTYPE b, DTYPE* h, DTYPE* s, DTYPE* v)
{
    DTYPE minc = fmin(r, fmin(g, b));
    DTYPE maxc = fmax(r, fmax(g, b));

    *v = maxc;

    DTYPE cr = maxc - minc;
    bool eqc = (cr == 0);

    *s = cr / (eqc ? (DTYPE)1.0 : maxc);

    DTYPE cr_divisor = eqc ? (DTYPE)1.0 : cr;
    DTYPE rc         = (maxc - r) / cr_divisor;
    DTYPE gc         = (maxc - g) / cr_divisor;
    DTYPE bc         = (maxc - b) / cr_divisor;

    DTYPE hr = (maxc == r) * (bc - gc);
    DTYPE hg = ((maxc == g) & (maxc != r)) * ((DTYPE)2.0 + rc - bc);
    DTYPE hb = ((maxc != g) & (maxc != r)) * ((DTYPE)4.0 + gc - rc);

    *h = fmod((hr + hg + hb) / (DTYPE)6.0 + (DTYPE)1.0, (DTYPE)1.0);
}

template <typename DTYPE>
__device__ void
convertHSVToRGB(const DTYPE h, const DTYPE s, const DTYPE v, DTYPE* r, DTYPE* g, DTYPE* b)
{
    DTYPE i    = floor(h * (DTYPE)6.0);
    DTYPE f    = (h * (DTYPE)6.0) - i;
    int i_case = ((int)i + 6) % 6;

    DTYPE p = clamp(v * ((DTYPE)1.0 - s), (DTYPE)0.0, (DTYPE)1.0);
    DTYPE q = clamp(v * ((DTYPE)1.0 - s * f), (DTYPE)0.0, (DTYPE)1.0);
    DTYPE t = clamp(v * ((DTYPE)1.0 - s * ((DTYPE)1.0 - f)), (DTYPE)0.0, (DTYPE)1.0);

    switch(i_case)
    {
    case 0:
        *r = v;
        *g = t;
        *b = p;
        break;
    case 1:
        *r = q;
        *g = v;
        *b = p;
        break;
    case 2:
        *r = p;
        *g = v;
        *b = t;
        break;
    case 3:
        *r = p;
        *g = q;
        *b = v;
        break;
    case 4:
        *r = t;
        *g = p;
        *b = v;
        break;
    case 5:
        *r = v;
        *g = p;
        *b = q;
        break;
    default:
        // Achievement Get: How Did We Get Here?
        return;
    }
}

template <typename DTYPE>
__device__ void DeviceImageAdjustHue(const DTYPE* __restrict__ input,
                                     DTYPE* __restrict__ output,
                                     const float hue_factor,
                                     const size_t N,
                                     const size_t /* c_stride */,
                                     const tensor_view_4d_t input_tv,
                                     const tensor_view_4d_t output_tv)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= N)
        return;

    int n, c, h, w;
    getNCHW(n, c, h, w, gid, input_tv.size);

    n = n * 3 + c;

    DTYPE r = get4DValueAt(input, input_tv.stride, n, 0, h, w);
    DTYPE g = get4DValueAt(input, input_tv.stride, n, 1, h, w);
    DTYPE b = get4DValueAt(input, input_tv.stride, n, 2, h, w);

    FLOAT_ACCUM fr = CVT_FLOAT2ACCUM(r);
    FLOAT_ACCUM fg = CVT_FLOAT2ACCUM(g);
    FLOAT_ACCUM fb = CVT_FLOAT2ACCUM(b);

    FLOAT_ACCUM hue, sat, val;
    convertRGBToHSV(fr, fg, fb, &hue, &sat, &val);
    hue = fmod(hue + hue_factor, 1.0f);
    convertHSVToRGB(hue, sat, val, &fr, &fg, &fb);

    set4DValueAt(output, output_tv, n, 0, h, w, CVT_ACCUM2FLOAT(fr));
    set4DValueAt(output, output_tv, n, 1, h, w, CVT_ACCUM2FLOAT(fg));
    set4DValueAt(output, output_tv, n, 2, h, w, CVT_ACCUM2FLOAT(fb));
}

template <typename DTYPE>
__device__ void DeviceImageAdjustHueContiguous(const DTYPE* __restrict__ input,
                                               DTYPE* __restrict__ output,
                                               const float hue_factor,
                                               const size_t N,
                                               const size_t c_stride,
                                               const size_t input_off,
                                               const size_t output_off)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= N)
        return;

    const size_t n   = gid / c_stride;
    const size_t idx = gid % c_stride;

    size_t pixel_idx = idx + n * c_stride * 3;

    DTYPE r = input[pixel_idx + input_off];
    DTYPE g = input[pixel_idx + c_stride + input_off];
    DTYPE b = input[pixel_idx + c_stride * 2 + input_off];

    FLOAT_ACCUM fr = CVT_FLOAT2ACCUM(r);
    FLOAT_ACCUM fg = CVT_FLOAT2ACCUM(g);
    FLOAT_ACCUM fb = CVT_FLOAT2ACCUM(b);

    FLOAT_ACCUM hue, sat, val;

    convertRGBToHSV(fr, fg, fb, &hue, &sat, &val);
    hue = fmod(hue + hue_factor, 1.0f);
    convertHSVToRGB(hue, sat, val, &fr, &fg, &fb);

    output[pixel_idx + output_off]                = CVT_ACCUM2FLOAT(fr);
    output[pixel_idx + c_stride + output_off]     = CVT_ACCUM2FLOAT(fg);
    output[pixel_idx + c_stride * 2 + output_off] = CVT_ACCUM2FLOAT(fb);
}

template <typename DTYPE>
__device__ void DeviceImageAdjustHueNHWCContiguous(const DTYPE* __restrict__ input,
                                                   DTYPE* __restrict__ output,
                                                   const float hue_factor,
                                                   const size_t N,
                                                   const size_t /* c_stride */,
                                                   const size_t input_off,
                                                   const size_t output_off)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= N)
        return;

    DTYPE r = input[gid * 3 + input_off];
    DTYPE g = input[gid * 3 + 1 + input_off];
    DTYPE b = input[gid * 3 + 2 + input_off];

    FLOAT_ACCUM fr = CVT_FLOAT2ACCUM(r);
    FLOAT_ACCUM fg = CVT_FLOAT2ACCUM(g);
    FLOAT_ACCUM fb = CVT_FLOAT2ACCUM(b);

    FLOAT_ACCUM hue, sat, val;
    convertRGBToHSV(fr, fg, fb, &hue, &sat, &val);

    hue = fmod(hue + hue_factor, 1.0f);

    convertHSVToRGB(hue, sat, val, &fr, &fg, &fb);

    output[gid * 3 + output_off]     = CVT_ACCUM2FLOAT(fr);
    output[gid * 3 + 1 + output_off] = CVT_ACCUM2FLOAT(fg);
    output[gid * 3 + 2 + output_off] = CVT_ACCUM2FLOAT(fb);
}

// Trampolines
extern "C" __global__ void ImageAdjustHue(const FLOAT* __restrict__ input,
                                          FLOAT* __restrict__ output,
                                          const float hue_factor,
                                          const size_t N,
                                          const size_t c_stride,
                                          const tensor_view_4d_t input_tv,
                                          const tensor_view_4d_t output_tv)
{
    DeviceImageAdjustHue<FLOAT>(input, output, hue_factor, N, c_stride, input_tv, output_tv);
}

extern "C" __global__ void ImageAdjustHueContiguous(const FLOAT* __restrict__ input,
                                                    FLOAT* __restrict__ output,
                                                    const float hue_factor,
                                                    const size_t N,
                                                    const size_t c_stride,
                                                    const size_t input_off,
                                                    const size_t output_off)
{
    DeviceImageAdjustHueContiguous<FLOAT>(
        input, output, hue_factor, N, c_stride, input_off, output_off);
}
