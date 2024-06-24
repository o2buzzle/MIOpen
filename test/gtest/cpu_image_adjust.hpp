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

#ifndef TEST_GTEST_CPU_IMAGE_ADJUST_HPP
#define TEST_GTEST_CPU_IMAGE_ADJUST_HPP

#include "miopen/tensor_view.hpp"
#include "tensor_holder.hpp"

template <typename T>
T clamp(T val, T min, T max)
{
    val = val < min ? min : val;
    val = val > max ? max : val;
    return val;
}

template <typename T = float>
void mloConvertRGBToHSV(const T r, const T g, const T b, T* h, T* s, T* v)
{
    T minc = std::min(r, std::min(g, b));
    T maxc = std::max(r, std::max(g, b));

    *v = maxc;

    T cr     = maxc - minc;
    bool eqc = (cr == 0);

    *s = cr / (eqc ? 1.0f : maxc);

    T cr_divisor = eqc ? (T)1.0f : cr;
    T rc         = (maxc - r) / cr_divisor;
    T gc         = (maxc - g) / cr_divisor;
    T bc         = (maxc - b) / cr_divisor;

    T hr = static_cast<T>((maxc == r) * (bc - gc));
    T hg = static_cast<T>(((maxc == g) && (maxc != r)) * ((T)2.0f + rc - bc));
    T hb = static_cast<T>(((maxc != g) && (maxc != r)) * ((T)4.0f + gc - rc));

    *h = fmod((hr + hg + hb) / 6.0 + 1.0, 1.0);
}

template <typename T = float>
void mloConvertHSVToRGB(const T h, const T s, const T v, T* r, T* g, T* b)
{
    T i        = static_cast<T>(floor(h * 6.0));
    T f        = static_cast<T>(h * 6.0 - i);
    int i_case = ((int)i + 6) % 6;

    T p = static_cast<T>(clamp(v * (1.0f - s), 0.0f, 1.0f));
    T q = static_cast<T>(clamp(v * (1.0f - s * f), 0.0f, 1.0f));
    T t = static_cast<T>(clamp(v * (1.0f - s * (1.0f - f)), 0.0f, 1.0f));

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
        // This case should never happen (i_case is guaranteed to be in range [0,5])
        // Just in case this ever does, panic immediately
        MIOPEN_THROW("i_case out of range");
        break;
    }
}

template <typename T>
void mloRunImageAdjustHueHost(T* input_buf,
                              T* output_buf,
                              miopen::TensorDescriptor inputTensorDesc,
                              miopen::TensorDescriptor outputTensorDesc,
                              float hue_factor)
{
    size_t N       = inputTensorDesc.GetElementSize() / 3;
    auto input_tv  = get_inner_expanded_4d_tv(inputTensorDesc);
    auto output_tv = get_inner_expanded_4d_tv(outputTensorDesc);

    int n, c, h, w;
    for(auto gid = 0; gid < N; gid++)
    {
        getNCHW(n, c, h, w, gid, input_tv.size);

        n = n * 3 + c;

        T r = static_cast<T>(get4DValueAt(input_buf, input_tv, n, 0, h, w));
        T g = static_cast<T>(get4DValueAt(input_buf, input_tv, n, 1, h, w));
        T b = static_cast<T>(get4DValueAt(input_buf, input_tv, n, 2, h, w));

        T hue, sat, val;

        mloConvertRGBToHSV(r, g, b, &hue, &sat, &val);
        hue = fmod(hue + hue_factor, 1.0);
        mloConvertHSVToRGB(hue, sat, val, &r, &g, &b);

        set4DValueAt(output_buf, output_tv, n, 0, h, w, r);
        set4DValueAt(output_buf, output_tv, n, 1, h, w, g);
        set4DValueAt(output_buf, output_tv, n, 2, h, w, b);
    }
}

template <typename T>
void cpu_image_adjust_hue(tensor<T> input, tensor<T>& output, float hue)
{
    mloRunImageAdjustHueHost(input.data.data(), output.data.data(), input.desc, output.desc, hue);
}

template <typename T>
void mloImageAdjustBrightnessRunHost(const T* input,
                                     T* output,
                                     miopen::TensorDescriptor inputDesc,
                                     miopen::TensorDescriptor outputDesc,
                                     const float brightness_factor)
{
    tensor_view_4d_t input_tv  = get_inner_expanded_4d_tv(inputDesc);
    tensor_view_4d_t output_tv = get_inner_expanded_4d_tv(outputDesc);

    size_t N = inputDesc.GetElementSize();

    for(size_t gid = 0; gid < N; gid++)
    {
        T pixel  = get4DValueAt(input, input_tv, gid);
        T result = static_cast<T>(clamp(static_cast<float>(pixel) * brightness_factor, 0.0f, 1.0f));
        set4DValueAt(output, output_tv, gid, result);
    }
}

template <typename T>
void cpu_image_adjust_brightness(tensor<T> input, tensor<T>& output, float brightness)
{
    mloImageAdjustBrightnessRunHost(
        input.data.data(), output.data.data(), input.desc, output.desc, brightness);
}

template <typename T>
void RGBToGrayscale(const T* src,
                    T* dst,
                    const tensor_view_4d_t src_tv,
                    const tensor_view_4d_t dst_tv,
                    const size_t N)
{
    for(size_t gid = 0; gid < N; gid++)
    {
        int n, c, h, w;
        getNCHW(n, c, h, w, gid, dst_tv.size);

        T r = get4DValueAt(src, src_tv, n, 0, h, w);
        T g = get4DValueAt(src, src_tv, n, 1, h, w);
        T b = get4DValueAt(src, src_tv, n, 2, h, w);

        T value = static_cast<T>(0.2989 * r + 0.587 * g + 0.114 * b);

        // We expect the workspace here to always stay contiguous
        dst[dst_tv.offset + gid] = value;
    }
}

template <typename T>
void Blend(const T* img1,
           const T* img2,
           T* output,
           const tensor_view_4d_t img1_tv,
           const tensor_view_4d_t img2_tv,
           const tensor_view_4d_t output_tv,
           const size_t n_stride,
           const size_t c_stride,
           const size_t N,
           float ratio,
           float bound)

{
    for(size_t gid = 0; gid < N; gid++)
    {
        const size_t n        = gid / n_stride;
        const size_t img2_idx = n * c_stride + gid % c_stride;

        T img1_v = get4DValueAt(img1, img1_tv, gid);
        T img2_v = img2[img2_tv.offset + img2_idx];

        T result = static_cast<T>(clamp((ratio * img1_v + (1.0f - ratio) * img2_v), 0.0f, bound));

        set4DValueAt(output, output_tv, gid, result);
    }
}

template <typename T>
void mloImageAdjustSaturationRunHost(miopen::TensorDescriptor inputDesc,
                                     miopen::TensorDescriptor outputDesc,
                                     const T* input,
                                     T* output,
                                     float saturation_factor)

{
    tensor_view_4d_t input_tv  = get_inner_expanded_4d_tv(inputDesc);
    tensor_view_4d_t output_tv = get_inner_expanded_4d_tv(outputDesc);

    // temporary view for workspace (basically a contiguous vector with same size as input_tv)
    std::vector<T> workspace = std::vector<T>(inputDesc.GetElementSize(), static_cast<T>(0.0f));
    miopen::TensorDescriptor wsDesc =
        miopen::TensorDescriptor{inputDesc.GetType(), inputDesc.GetLengths()};

    auto ws_tv = get_inner_expanded_4d_tv(wsDesc);

    auto N        = inputDesc.GetElementSize();
    auto c_stride = input_tv.size[2] * input_tv.size[3];
    auto n_stride = c_stride * input_tv.size[1];

    float bound = 1.0f;

    RGBToGrayscale(input, workspace.data(), input_tv, ws_tv, N);
    Blend(input,
          workspace.data(),
          output,
          input_tv,
          ws_tv,
          output_tv,
          n_stride,
          c_stride,
          N,
          saturation_factor,
          bound);
}

template <typename T>
void cpu_image_adjust_saturation(tensor<T> input, tensor<T>& output, float saturation_factor)
{
    mloImageAdjustSaturationRunHost(
        input.desc, output.desc, input.data.data(), output.data.data(), saturation_factor);
}

template <typename T>
void mloImageNormalizeRunHost(miopen::TensorDescriptor inputDesc,
                              miopen::TensorDescriptor outputDesc,
                              miopen::TensorDescriptor meanDesc,
                              miopen::TensorDescriptor stdvarDesc,
                              const T* input,
                              T* output,
                              const T* mean,
                              const T* stdvar)
{
    tensor_view_4d_t input_tv  = get_inner_expanded_4d_tv(inputDesc);
    tensor_view_4d_t output_tv = get_inner_expanded_4d_tv(outputDesc);
    tensor_view_4d_t mean_tv   = get_inner_expanded_4d_tv(meanDesc);
    tensor_view_4d_t stdvar_tv = get_inner_expanded_4d_tv(stdvarDesc);

    auto N         = inputDesc.GetElementSize();
    auto C         = input_tv.size[1];
    auto c_strides = input_tv.stride[1];

    for(size_t gid = 0; gid < N; gid++)
    {
        auto c = gid / c_strides % C;

        T pixel  = get4DValueAt(input, input_tv, gid);
        T result = (pixel - static_cast<T>(mean[c + mean_tv.offset])) /
                   static_cast<T>(stdvar[c + stdvar_tv.offset]);
        set4DValueAt(output, output_tv, gid, result);
    }
}

template <typename T>
void cpu_image_normalize(tensor<T> input, tensor<T>& output, tensor<T> mean, tensor<T> stdvar)
{
    mloImageNormalizeRunHost(input.desc,
                             output.desc,
                             mean.desc,
                             stdvar.desc,
                             input.data.data(),
                             output.data.data(),
                             mean.data.data(),
                             stdvar.data.data());
}

#endif
