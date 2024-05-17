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

#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>
#include <miopen/mseloss.hpp>

extern "C" miopenStatus_t miopenMSELossForward(miopenHandle_t handle,
                                               miopenTensorDescriptor_t xDesc,
                                               miopenTensorDescriptor_t yDesc,
                                               miopenTensorDescriptor_t zDesc,
                                               const void* x,
                                               const void* y,
                                               void* z,
                                               const float lossScale)
{
    MIOPEN_LOG_FUNCTION(xDesc, yDesc, zDesc, x, y, z, lossScale);

    return miopen::try_([&] {
        miopen::miopenMSELossForward(miopen::deref(handle),
                                     miopen::deref(xDesc),
                                     miopen::deref(yDesc),
                                     miopen::deref(zDesc),
                                     DataCast(x),
                                     DataCast(y),
                                     DataCast(z),
                                     lossScale);
    });
}

extern "C" miopenStatus_t miopenMSELossBackward(miopenHandle_t handle,
                                                miopenTensorDescriptor_t xDesc,
                                                miopenTensorDescriptor_t yDesc,
                                                miopenTensorDescriptor_t dzDesc,
                                                miopenTensorDescriptor_t dxDesc,
                                                miopenTensorDescriptor_t dyDesc,
                                                const void* x,
                                                const void* y,
                                                const void* dz,
                                                void* dx,
                                                void* dy,
                                                const float lossScale)
{
    MIOPEN_LOG_FUNCTION(xDesc, yDesc, dzDesc, dxDesc, dyDesc, x, y, dz, dx, dy, lossScale);

    return miopen::try_([&] {
        miopen::miopenMSELossBackward(miopen::deref(handle),
                                      miopen::deref(xDesc),
                                      miopen::deref(yDesc),
                                      miopen::deref(dzDesc),
                                      miopen::deref(dxDesc),
                                      miopen::deref(dyDesc),
                                      DataCast(x),
                                      DataCast(y),
                                      DataCast(dz),
                                      DataCast(dx),
                                      DataCast(dy),
                                      lossScale);
    });
}
