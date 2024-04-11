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

/*
>>> m = nn.ConstantPad2d(2, 3.5)
>>> input = torch.randn(1, 2, 2)
>>> input
tensor([[[ 1.6585,  0.4320],
         [-0.8701, -0.4649]]])
>>> m(input)
tensor([[[ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000,  3.5000],
         [ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000,  3.5000],
         [ 3.5000,  3.5000,  1.6585,  0.4320,  3.5000,  3.5000],
         [ 3.5000,  3.5000, -0.8701, -0.4649,  3.5000,  3.5000],
         [ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000,  3.5000],
         [ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000,  3.5000]]])
>>> # using different paddings for different sides
>>> m = nn.ConstantPad2d((3, 0, 2, 1), 3.5)
>>> m(input)
tensor([[[ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000],
         [ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000],
         [ 3.5000,  3.5000,  3.5000,  1.6585,  0.4320],
         [ 3.5000,  3.5000,  3.5000, -0.8701, -0.4649],
         [ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000]]])
*/

#ifndef GUARD_CPU_PAD_CONSTANT_HPP
#define GUARD_CPU_PAD_CONSTANT_HPP

#include "tensor_holder.hpp"

template <typename T>
T get5DValueAt(const T* x, const size_t* x_dims, size_t n, size_t c, size_t d, size_t h, size_t w)
{
    return x[n * x_dims[1] * x_dims[2] * x_dims[3] * x_dims[4] +
             c * x_dims[2] * x_dims[3] * x_dims[4] + d * x_dims[3] * x_dims[4] + h * x_dims[4] + w];
}

template <class T>
void PadConstantForward(tensor<T> input, tensor<T> ref_output, const int padding[], float padding_value)

{
    auto ref_output_dims = ref_output.desc.GetLengths();
    auto input_dims      = input.desc.GetLengths();

    int o[5];

    for(auto gid = ref_output.begin(); gid != ref_output.end(); ++gid)
    {
        bool flag = true;

        ulong ncdh = gid / ref_output_dims[4];
        o[4]       = gid % ref_output_dims[4];
        ulong ncd  = ncdh / ref_output_dims[3];
        o[3]       = ncdh % ref_output_dims[3];
        ulong nc   = ncd / ref_output_dims[2];
        o[2]       = ncd % ref_output_dims[2];
        o[1]       = nc / ref_output_dims[1];
        o[0]       = nc % ref_output_dims[1];

        for(int i = 0; i < 5; i++)
        {
            o[i] = o[i] - padding[2 * i];
            flag *= (o[i] >= 0 && o[i] < ref_output_dims[i]);
        }

        if(flag)
        {
            // This value should be copied from the input tensor
            ref_output[gid] = get5DValueAt(input.data, input_dims, o[0], o[1], o[2], o[3], o[4]);
        }
        else
        {
            ref_output[gid] = padding_value;
        }
    }
    // how much do you wanna bet on this instantly blowing up?
}

#endif
