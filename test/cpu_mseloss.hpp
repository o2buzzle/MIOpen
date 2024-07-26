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

#ifndef GUARD_CPU_MSE_LOSS_HPP
#define GUARD_CPU_MSE_LOSS_HPP

#include "miopen/tensor.hpp"
#include "miopen/tensor_view_utils.hpp"

template <class T>
void cpu_mseloss(miopen::TensorDescriptor inputDesc,
                 miopen::TensorDescriptor targetDesc,
                 miopen::TensorDescriptor outputDesc,
                 const T* input,
                 const T* target,
                 T* output,
                 float divisor)
{
    const int local_size = 256;

    tensor_view_t<5> I_tv = miopen::get_inner_expanded_tv<5>(inputDesc);
    tensor_view_t<5> T_tv = miopen::get_inner_expanded_tv<5>(targetDesc);

    int64_t gid = 0;
    auto size   = inputDesc.GetElementSize();
    auto ref_workspace =
        std::vector<T>(size + ((size / local_size) + 1) * local_size, static_cast<T>(0.0f));

    while(true)
    {
        size_t n0123 = gid / I_tv.size[4], n4 = gid % I_tv.size[4];
        size_t n012 = n0123 / I_tv.size[3], n3 = n0123 % I_tv.size[3];
        size_t n01 = n012 / I_tv.size[2], n2 = n012 % I_tv.size[2];
        size_t n0 = n01 / I_tv.size[1], n1 = n01 % I_tv.size[1];

        if(!(n0 < I_tv.size[0]))
            break;

        // size_t Iidx = get5DIndexAt<size_t>(I_tv, n0, n1, n2, n3, n4);
        // size_t Tidx = get5DIndexAt<size_t>(T_tv, n0, n1, n2, n3, n4);

        size_t Iidx = I_tv.get_tensor_view_idx({n0, n1, n2, n3, n4});
        size_t Tidx = T_tv.get_tensor_view_idx({n0, n1, n2, n3, n4});

        ref_workspace[gid] =
            static_cast<T>((input[Iidx] - target[Tidx]) * (input[Iidx] - target[Tidx]) / divisor);

        ++gid;
    }

    // Yes this mess is actually necessary to emulate the behavior of parallel reduction.
    // Naive approach here would generate __way too much__ floating point differences
    int offset_a = 0;
    int offset_b = size;
    size_t _size = size;
    do
    {
        for(int i = 0; i < _size; i += local_size)
        {
            T shared[local_size];
            for(int j = 0; j < local_size; ++j)
                shared[j] = i + j < _size ? ref_workspace[offset_a + i + j] : 0.0f;
            for(int offset = local_size / 2; offset > 0; offset >>= 1)
                for(int j = 0; j < offset; ++j)
                    shared[j] += shared[j + offset];
            if(_size <= local_size)
                output[0] = shared[0];
            else
                ref_workspace[offset_b + i / local_size] = shared[0];
        }
        std::swap(offset_a, offset_b);
        _size = (_size + local_size - 1) / local_size;
    } while(_size > 1);
}

template <class T>
void cpu_mseloss_backward(miopen::TensorDescriptor inputDesc,
                          miopen::TensorDescriptor targetDesc,
                          miopen::TensorDescriptor outputDesc,
                          miopen::TensorDescriptor inputGradDesc,
                          miopen::TensorDescriptor targetGradDesc,
                          const T* input,
                          const T* target,
                          const T* output,
                          T* input_grad,
                          T* target_grad,
                          float divisor)
{
    tensor_view_t<5> I_tv  = miopen::get_inner_expanded_tv<5>(inputDesc);
    tensor_view_t<5> T_tv  = miopen::get_inner_expanded_tv<5>(targetDesc);
    tensor_view_t<5> IG_tv = miopen::get_inner_expanded_tv<5>(inputGradDesc);
    tensor_view_t<5> TG_tv = miopen::get_inner_expanded_tv<5>(targetGradDesc);

    int64_t gid = 0;

    while(true)
    {
        size_t n0123 = gid / I_tv.size[4], n4 = gid % I_tv.size[4];
        size_t n012 = n0123 / I_tv.size[3], n3 = n0123 % I_tv.size[3];
        size_t n01 = n012 / I_tv.size[2], n2 = n012 % I_tv.size[2];
        size_t n0 = n01 / I_tv.size[1], n1 = n01 % I_tv.size[1];

        if(!(n0 < I_tv.size[0]))
            break;

        // size_t Iidx  = get5DIndexAt<size_t>(I_tv, n0, n1, n2, n3, n4);
        // size_t Tidx  = get5DIndexAt<size_t>(T_tv, n0, n1, n2, n3, n4);
        // size_t IGidx = get5DIndexAt<size_t>(IG_tv, n0, n1, n2, n3, n4);
        // size_t TGidx = get5DIndexAt<size_t>(TG_tv, n0, n1, n2, n3, n4);

        size_t Iidx  = I_tv.get_tensor_view_idx({n0, n1, n2, n3, n4});
        size_t Tidx  = T_tv.get_tensor_view_idx({n0, n1, n2, n3, n4});
        size_t IGidx = IG_tv.get_tensor_view_idx({n0, n1, n2, n3, n4});
        size_t TGidx = TG_tv.get_tensor_view_idx({n0, n1, n2, n3, n4});

        T grad = static_cast<T>(2.0f) * (input[Iidx] - target[Tidx]) / static_cast<T>(divisor) *
                 output[0];

        if(input_grad != nullptr)
        {
            input_grad[IGidx] = grad;
        }

        if(target_grad != nullptr)
        {
            target_grad[TGidx] = -grad;
        }
        ++gid;
    }
}
#endif
