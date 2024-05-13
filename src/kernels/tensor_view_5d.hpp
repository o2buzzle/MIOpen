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

#ifndef GUARD_TENSOR_VIEW_H
#define GUARD_TENSOR_VIEW_H

#include <hip/hip_runtime.h>

struct tensor_view_5d_t
{
    uint64_t size[5];
    uint64_t stride[5];
};

struct padding_5d_t
{
    uint64_t val[10];
};

template <typename T>
__host__ __device__ void inline getNCDHW(T* ncdhw, const T idx, const T size[5])
{
    ulong ncdh = (idx) / size[4];
    ncdhw[4]   = (idx) % size[4];
    ulong ncd  = ncdh / size[3];
    ncdhw[3]   = ncdh % size[3];
    ulong nc   = ncd / size[2];
    ncdhw[2]   = ncd % size[2];
    ncdhw[0]   = nc / size[1];
    ncdhw[1]   = nc % size[1];
}

template <typename T, typename U>
__host__
    __device__ T inline get5DValueAt(const T* x, const uint64_t* x_strides, U n, U c, U d, U h, U w)
{
    return x[n * x_strides[0] + c * x_strides[1] + d * x_strides[2] + h * x_strides[3] +
             w * x_strides[4]];
}

template <typename T>
__host__ __device__ void inline set5DValueAt(T* x, const tensor_view_5d_t& x_tv, size_t idx, T val)
{
    uint64_t o[5];
    o[4] = x_tv.stride[0] *
           (size_t)((idx) / x_tv.size[4] / x_tv.size[3] / x_tv.size[2] / x_tv.size[1]);
    o[3] = x_tv.stride[1] *
           ((size_t)((idx) / x_tv.size[4] / x_tv.size[3] / x_tv.size[2]) % x_tv.size[1]);
    o[2] = x_tv.stride[2] * ((size_t)((idx) / x_tv.size[4] / x_tv.size[3]) % x_tv.size[2]);
    o[1] = x_tv.stride[3] * ((size_t)((idx) / x_tv.size[4]) % x_tv.size[3]);
    o[0] = x_tv.stride[4] * ((idx) % x_tv.size[4]);
    x[o[0] + o[1] + o[2] + o[3] + o[4]] = val;
}

#endif
