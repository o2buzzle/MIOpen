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

    T hr = (maxc == r) * (bc - gc);
    T hg = ((maxc == g) && (maxc != r)) * ((T)2.0f + rc - bc);
    T hb = ((maxc != g) && (maxc != r)) * ((T)4.0f + gc - rc);

    *h = fmod((hr + hg + hb) / 6.0 + 1.0, 1.0);
}

template <typename T = float>
void mloConvertHSVToRGB(const T h, const T s, const T v, T* r, T* g, T* b)
{
    T i        = floor(h * 6.0);
    T f        = h * 6.0 - i;
    int i_case = ((int)i + 6) % 6;

    T p = clamp(v * (1.0 - s), 0.0, 1.0);
    T q = clamp(v * (1.0 - s * f), 0.0, 1.0);
    T t = clamp(v * (1.0 - s * (1.0 - f)), 0.0, 1.0);

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
        printf("i_case = %d\n", i_case);
        assert(false);
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
        T result = clamp(pixel * brightness_factor, T(0.0f), T(1.0f));
        set4DValueAt(output, output_tv, gid, result);
    }
}

template <typename T>
void cpu_image_adjust_brightness(tensor<T> input, tensor<T>& output, float brightness)
{
    mloImageAdjustBrightnessRunHost(
        input.data.data(), output.data.data(), input.desc, output.desc, brightness);
}

#endif
