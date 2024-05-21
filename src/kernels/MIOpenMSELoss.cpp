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

#include <cstddef>
#include <cstdio>
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"
#include "tensor_view.hpp"

#ifndef MIOPEN_USE_BFP16
#define MIOPEN_USE_BFP16 1
#define CVT_ACCUM2FLOAT(x) (float_to_bfloat16(x))
#endif

#ifndef IO_TYPE
#define IO_TYPE float
#endif

__device__ void DeviceMSELossForward5d(const IO_TYPE* __restrict__ I,
                                       const IO_TYPE* __restrict__ T,
                                       IO_TYPE* __restrict__ lsum,
                                       FLOAT_ACCUM divisor,
                                       tensor_view_5d_t I_tv,
                                       tensor_view_5d_t T_tv)
{
    divisor = CVT_ACCUM2FLOAT(divisor);

    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    size_t n0123 = gid / I_tv.size[4], n4 = gid % I_tv.size[4];
    size_t n012 = n0123 / I_tv.size[3], n3 = n0123 % I_tv.size[3];
    size_t n01 = n012 / I_tv.size[2], n2 = n012 % I_tv.size[2];
    size_t n0 = n01 / I_tv.size[1], n1 = n01 % I_tv.size[1];

    if(!(n0 < I_tv.size[0]))
        return;

    size_t Iidx = get5DIndexAt<size_t>(I_tv, n0, n1, n2, n3, n4);
    size_t Tidx = get5DIndexAt<size_t>(T_tv, n0, n1, n2, n3, n4);

    lsum[gid] = (I[Iidx] - T[Tidx]) * (I[Iidx] - T[Tidx]);
    lsum[gid] /= divisor;
}

__device__ void DeviceMSELossBackward5d(const IO_TYPE* __restrict__ I,
                                        const IO_TYPE* __restrict__ T,
                                        const IO_TYPE* __restrict__ dO,
                                        IO_TYPE* __restrict__ dI,
                                        IO_TYPE* __restrict__ dT,
                                        FLOAT_ACCUM divisor,
                                        tensor_view_5d_t I_tv,
                                        tensor_view_5d_t T_tv,
                                        tensor_view_5d_t dO_tv,
                                        tensor_view_5d_t dI_tv,
                                        tensor_view_5d_t dT_tv)
{
    const size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n0123 = gid / I_tv.size[4], n4 = gid % I_tv.size[4];
    size_t n012 = n0123 / I_tv.size[3], n3 = n0123 % I_tv.size[3];
    size_t n01 = n012 / I_tv.size[2], n2 = n012 % I_tv.size[2];
    size_t n0 = n01 / I_tv.size[1], n1 = n01 % I_tv.size[1];

    if(!(n0 < I_tv.size[0]))
        return;

    size_t Iidx  = get5DIndexAt<size_t>(I_tv, n0, n1, n2, n3, n4);
    size_t Tidx  = get5DIndexAt<size_t>(T_tv, n0, n1, n2, n3, n4);
    IO_TYPE grad = 0.0;
    grad         = 2.0f * (I[Iidx] - T[Tidx]) / divisor * dO[dO_tv.offset];

    if(dI != nullptr)
    {
        size_t dIidx = get5DIndexAt<size_t>(dI_tv, n0, n1, n2, n3, n4);

        dI[dIidx] = grad;
    }
    if(dT != nullptr)
    {
        size_t dTidx = get5DIndexAt<size_t>(dT_tv, n0, n1, n2, n3, n4);

        dT[dTidx] = -grad;
    }
}

__device__ void DeviceMSELossUnreducedForward5d(const IO_TYPE* __restrict__ I,
                                                const IO_TYPE* __restrict__ T,
                                                IO_TYPE* __restrict__ O,
                                                tensor_view_5d_t I_tv,
                                                tensor_view_5d_t T_tv,
                                                tensor_view_5d_t O_tv)
{
    const size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    size_t n0123 = gid / I_tv.size[4], n4 = gid % I_tv.size[4];
    size_t n012 = n0123 / I_tv.size[3], n3 = n0123 % I_tv.size[3];
    size_t n01 = n012 / I_tv.size[2], n2 = n012 % I_tv.size[2];
    size_t n0 = n01 / I_tv.size[1], n1 = n01 % I_tv.size[1];

    if(!(n0 < I_tv.size[0]))
        return;

    size_t Iidx = get5DIndexAt<size_t>(I_tv, n0, n1, n2, n3, n4);
    size_t Tidx = get5DIndexAt<size_t>(T_tv, n0, n1, n2, n3, n4);
    size_t Oidx = get5DIndexAt<size_t>(O_tv, n0, n1, n2, n3, n4);

    O[Oidx] = (I[Iidx] - T[Tidx]) * (I[Iidx] - T[Tidx]);
}

__device__ void DeviceMSELossUnreducedBackward5d(const IO_TYPE* __restrict__ I,
                                                 const IO_TYPE* __restrict__ T,
                                                 const IO_TYPE* __restrict__ dO,
                                                 IO_TYPE* __restrict__ dI,
                                                 IO_TYPE* __restrict__ dT,
                                                 tensor_view_5d_t I_tv,
                                                 tensor_view_5d_t T_tv,
                                                 tensor_view_5d_t dO_tv,
                                                 tensor_view_5d_t dI_tv,
                                                 tensor_view_5d_t dT_tv)
{
    const size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n0123 = gid / I_tv.size[4], n4 = gid % I_tv.size[4];
    size_t n012 = n0123 / I_tv.size[3], n3 = n0123 % I_tv.size[3];
    size_t n01 = n012 / I_tv.size[2], n2 = n012 % I_tv.size[2];
    size_t n0 = n01 / I_tv.size[1], n1 = n01 % I_tv.size[1];

    if(!(n0 < I_tv.size[0]))
        return;

    size_t Iidx  = get5DIndexAt<size_t>(I_tv, n0, n1, n2, n3, n4);
    size_t Tidx  = get5DIndexAt<size_t>(T_tv, n0, n1, n2, n3, n4);
    size_t dOidx = get5DIndexAt<size_t>(dO_tv, n0, n1, n2, n3, n4);

    IO_TYPE grad = 2.0f * (I[Iidx] - T[Tidx]) * dO[dOidx];

    if(dI != nullptr)
    {
        size_t dIidx = get5DIndexAt<size_t>(dI_tv, n0, n1, n2, n3, n4);
        dI[dIidx]    = grad;
    }
    if(dT != nullptr)
    {
        size_t dTidx = get5DIndexAt<size_t>(dT_tv, n0, n1, n2, n3, n4);
        dT[dTidx]    = -grad;
    }
}

// Trampolines
extern "C" __global__ void MSELossForward5d(const IO_TYPE* __restrict__ I,
                                            const IO_TYPE* __restrict__ T,
                                            IO_TYPE* __restrict__ lsum,
                                            FLOAT_ACCUM divisor,
                                            tensor_view_5d_t I_tv,
                                            tensor_view_5d_t T_tv)

{
    DeviceMSELossForward5d(I, T, lsum, divisor, I_tv, T_tv);
}

extern "C" __global__ void MSELossBackward5d(const IO_TYPE* __restrict__ I,
                                             const IO_TYPE* __restrict__ T,
                                             const IO_TYPE* __restrict__ dO,
                                             IO_TYPE* __restrict__ dI,
                                             IO_TYPE* __restrict__ dT,
                                             FLOAT_ACCUM divisor,
                                             tensor_view_5d_t I_tv,
                                             tensor_view_5d_t T_tv,
                                             tensor_view_5d_t dO_tv,
                                             tensor_view_5d_t dI_tv,
                                             tensor_view_5d_t dT_tv)
{
    DeviceMSELossBackward5d(I, T, dO, dI, dT, divisor, I_tv, T_tv, dO_tv, dI_tv, dT_tv);
}
extern "C" __global__ void MSELossForwardUnreduced5d(const IO_TYPE* __restrict__ I,
                                                     const IO_TYPE* __restrict__ T,
                                                     IO_TYPE* __restrict__ O,
                                                     tensor_view_5d_t I_tv,
                                                     tensor_view_5d_t T_tv,
                                                     tensor_view_5d_t O_tv)
{
    DeviceMSELossUnreducedForward5d(I, T, O, I_tv, T_tv, O_tv);
}

extern "C" __global__ void MSELossBackwardUnreduced5d(const IO_TYPE* __restrict__ I,
                                                      const IO_TYPE* __restrict__ T,
                                                      const IO_TYPE* __restrict__ dO,
                                                      IO_TYPE* __restrict__ dI,
                                                      IO_TYPE* __restrict__ dT,
                                                      tensor_view_5d_t I_tv,
                                                      tensor_view_5d_t T_tv,
                                                      tensor_view_5d_t dO_tv,
                                                      tensor_view_5d_t dI_tv,
                                                      tensor_view_5d_t dT_tv)
{
    DeviceMSELossUnreducedBackward5d(I, T, dO, dI, dT, I_tv, T_tv, dO_tv, dI_tv, dT_tv);
}