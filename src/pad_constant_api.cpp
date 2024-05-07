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
#include <miopen/pad_constant.hpp>

extern "C" miopenStatus_t miopenPadConstantFwd(miopenHandle_t handle,
                                               miopenTensorDescriptor_t xDesc,
                                               miopenTensorDescriptor_t yDesc,
                                               const void* x,
                                               void* y,
                                               const size_t* padding,
                                               float value)
{
    MIOPEN_LOG_FUNCTION(handle, xDesc, x, padding, value, y);

    return miopen::try_([&] {
        miopen::PadConstantForward(miopen::deref(handle),
                                   miopen::deref(xDesc),
                                   miopen::deref(yDesc),
                                   DataCast(x),
                                   DataCast(y),
                                   padding,
                                   value);
    });
}

extern "C" miopenStatus_t miopenPadConstantBwd(miopenHandle_t handle,
                                               miopenTensorDescriptor_t xDesc,
                                               miopenTensorDescriptor_t yDesc,
                                               void* x,
                                               const void* y,
                                               const size_t* padding)
{
    MIOPEN_LOG_FUNCTION(handle, xDesc, x, y, padding);

    return miopen::try_([&] {
        miopen::PadConstantBackward(miopen::deref(handle),
                                    miopen::deref(xDesc),
                                    miopen::deref(yDesc),
                                    DataCast(x),
                                    DataCast(y),
                                    padding);
    });
}
