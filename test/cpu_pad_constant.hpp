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

#ifndef GUARD_CPU_PAD_CONSTANT_HPP
#define GUARD_CPU_PAD_CONSTANT_HPP

#include "miopen/tensor.hpp"
#include <cstddef>
#include <sys/types.h>
#include "../src/kernels/tensor_view_5d.hpp"

template <class T>
void cpu_pad_constant_fwd(const T* input,
                          T* output,
                          miopen::TensorDescriptor* input_desc,
                          miopen::TensorDescriptor* output_desc,
                          const size_t padding[10],
                          float value)
{
    size_t o[5];

    for(size_t gid = 0; gid != output_desc->GetElementSize(); ++gid)
    {
        bool flag = true;

        getNCDHW(o, gid, output_desc->GetLengths().data());

        for(int i = 0; i < 5; i++)
        {
            o[i] = o[i] - padding[2 * i];
            flag *= (o[i] < input_desc->GetLengths()[i]);
        }

        output[gid] =
            flag
                ? get5DValueAt(input, input_desc->GetStrides().data(), o[0], o[1], o[2], o[3], o[4])
                : (T)value;
    }
}
#endif

template <class T>
void cpu_pad_consant_bwd(T* input_grad,
                         T* backward_output,
                         miopen::TensorDescriptor* backward_output_desc,
                         miopen::TensorDescriptor* input_grad_desc,
                         const size_t padding[10])
{
    // Setup the tensorView (for set5dValueAt)
    tensor_view_5d_t o_tv;
    for(int i = 0; i < 5; i++)
    {
        o_tv.size[i]   = backward_output_desc->GetLengths()[i];
        o_tv.stride[i] = backward_output_desc->GetStrides()[i];
    }

    size_t o[5];

    auto backward_output_dims    = backward_output_desc->GetLengths();
    auto backward_output_strides = backward_output_desc->GetStrides();

    auto input_grad_dims    = input_grad_desc->GetLengths();
    auto input_grad_strides = input_grad_desc->GetStrides();

    size_t backward_output_size = backward_output_desc->GetElementSize();

    for(size_t gid = 0; gid < backward_output_size; ++gid)
    {
        bool flag = true;
        getNCDHW(o, gid, backward_output_dims.data());

        for(int i = 0; i < 5; i++)
        {
            o[i] = o[i] + padding[2 * i];
            flag *= (o[i] < input_grad_dims[i]);
        }

        if(flag)
        {
            auto val = get5DValueAt<T>(
                input_grad, input_grad_strides.data(), o[0], o[1], o[2], o[3], o[4]);
            set5DValueAt(backward_output, o_tv, gid, val);
        }
    }
}
