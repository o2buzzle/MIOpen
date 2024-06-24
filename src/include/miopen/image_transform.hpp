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
#include "miopen/miopen.h"

#ifndef GUARD_MIOPEN_IMAGE_TRANSFORM_HPP_
#define GUARD_MIOPEN_IMAGE_TRANSFORM_HPP_

namespace miopen {
struct Handle;
struct TensorDescriptor;

miopenStatus_t ImageAdjustHue(Handle& handle,
                              const TensorDescriptor& inputTensorDesc,
                              const TensorDescriptor& outputTensorDesc,
                              ConstData_t input_buf,
                              Data_t output_buf,
                              float hue);

miopenStatus_t ImageAdjustBrightness(Handle& handle,
                                     const TensorDescriptor& inputTensorDesc,
                                     const TensorDescriptor& outputTensorDesc,
                                     ConstData_t input_buf,
                                     Data_t output_buf,
                                     float brightness_factor);

miopenStatus_t ImageNormalize(Handle& handle,
                              const TensorDescriptor& inputTensorDesc,
                              const TensorDescriptor& meanTensorDesc,
                              const TensorDescriptor& stdTensorDesc,
                              const TensorDescriptor& outputTensorDesc,
                              ConstData_t input_buf,
                              ConstData_t mean_buf,
                              ConstData_t std_buf,
                              Data_t output_buf);

miopenStatus_t ImageAdjustContrast(Handle& handle,
                                   const TensorDescriptor& inputTensorDesc,
                                   const TensorDescriptor& outputTensorDesc,
                                   ConstData_t input_buf,
                                   Data_t workspace_buf,
                                   Data_t output_buf,
                                   float contrast_factor);

size_t ImageAdjustContrastGetWorkspaceSize(Handle& handle,
                                           const TensorDescriptor& inputTensorDesc,
                                           const TensorDescriptor& outputTensorDesc);

miopenStatus_t ImageAdjustSaturation(Handle& handle,
                                     const TensorDescriptor& inputTensorDesc,
                                     const TensorDescriptor& outputTensorDesc,
                                     ConstData_t input_buf,
                                     Data_t workspace_buf,
                                     Data_t output_buf,
                                     float saturation_factor);

size_t ImageAdjustSaturationGetWorkspaceSize(Handle& handle,
                                             const TensorDescriptor& inputTensorDesc,
                                             const TensorDescriptor& outputTensorDesc,
                                             float saturation_factor);

} // namespace miopen

#endif
