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

#include "miopen/common.hpp"

#ifndef GUARD_MIOPEN_IMAGE_TRANSFORM_HPP_
#define GUARD_MIOPEN_IMAGE_TRANSFORM_HPP_

namespace miopen {
struct Handle;
struct TensorDescriptor;

miopenStatus_t miopenImageAdjustHue(Handle& handle,
                                    const TensorDescriptor& inputTensorDesc,
                                    const TensorDescriptor& outputTensorDesc,
                                    ConstData_t input_buf,
                                    Data_t output_buf,
                                    float hue);

miopenStatus_t miopenImageAdjustBrightness(Handle& handle,
                                           const TensorDescriptor& inputTensorDesc,
                                           const TensorDescriptor& outputTensorDesc,
                                           ConstData_t input_buf,
                                           Data_t output_buf,
                                           float brightness_factor);

miopenStatus_t miopenImageNormalize(Handle& handle,
                                    const TensorDescriptor& inputTensorDesc,
                                    const TensorDescriptor& meanTensorDesc,
                                    const TensorDescriptor& stdTensorDesc,
                                    const TensorDescriptor& outputTensorDesc,
                                    ConstData_t input_buf,
                                    ConstData_t mean_buf,
                                    ConstData_t std_buf,
                                    Data_t output_buf);

} // namespace miopen

#endif
