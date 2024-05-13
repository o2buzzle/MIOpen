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
                          miopen::TensorDescriptor* input_dims,
                          miopen::TensorDescriptor* output_dims,
                          const size_t padding[10],
                          float value)
{
    size_t o[5];

    for(size_t gid = 0; gid != output_dims->GetElementSize(); ++gid)
    {
        bool flag = true;

        getNCDHW(o, gid, output_dims->GetLengths().data());

        for(int i = 0; i < 5; i++)
        {
            o[i] = o[i] - padding[2 * i];
            flag *= (o[i] < input_dims->GetLengths()[i]);
        }

        output[gid] =
            flag
                ? get5DValueAt(input, input_dims->GetStrides().data(), o[0], o[1], o[2], o[3], o[4])
                : (T)value;
    }
}
#endif
