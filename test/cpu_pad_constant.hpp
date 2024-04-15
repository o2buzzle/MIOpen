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

#include <cstddef>
#include <sys/types.h>

template <typename T>
T get5DValueAt(const T* x, const size_t* x_dims, size_t n, size_t c, size_t d, size_t h, size_t w)
{
    return x[n * x_dims[1] * x_dims[2] * x_dims[3] * x_dims[4] +
             c * x_dims[2] * x_dims[3] * x_dims[4] + d * x_dims[3] * x_dims[4] + h * x_dims[4] + w];
}

#define GET_NCDHW(n, c, d, h, w, idx, size) \
    {                                       \
        ulong ncdh = (idx) / size[4];       \
        w          = (idx) % size[4];       \
        ulong ncd  = ncdh / size[3];        \
        h          = ncdh % size[3];        \
        ulong nc   = ncd / size[2];         \
        d          = ncd % size[2];         \
        n          = nc / size[1];          \
        c          = nc % size[1];          \
    }

template <class T>
void cpu_pad_constant_fwd(const T* input,
                          T* output,
                          const size_t* input_dims,
                          const size_t* output_dims,
                          const size_t* padding,
                          T value)
{
    size_t o[5];

    // printf("Tensor output size is reported to be n=%lu c=%lu d=%lu h=%lu w=%lu\n",
    //        output_dims[0],
    //        output_dims[1],
    //        output_dims[2],
    //        output_dims[3],
    //        output_dims[4]);

    for(size_t gid = 0;
        gid != output_dims[0] * output_dims[1] * output_dims[2] * output_dims[3] * output_dims[4];
        ++gid)
    {
        bool flag = true;

        GET_NCDHW(o[0], o[1], o[2], o[3], o[4], gid, output_dims);

        for(int i = 0; i < 5; i++)
        {
            o[i] = o[i] - padding[2 * i];
            flag *= (o[i] < input_dims[i]);
        }

        if(flag)
        {
            // This value should be copied from the input tensor
            // ref_output[gid] = get5DValueAt(input.data.data(), input_dims.data(), o[0], o[1],
            // o[2], o[3], o[4]);
            output[gid] = get5DValueAt(input, input_dims, o[0], o[1], o[2], o[3], o[4]);
        }
        else
        {
            // This value should be constant
            output[gid] = value;
        }
    }
    // how much do you wanna bet on this instantly blowing up?
}
#endif
