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

#ifndef MIOPEN_ROI_ALIGN_HPP_
#define MIOPEN_ROI_ALIGN_HPP_

#include <miopen/handle.hpp>
#include <miopen/tensor.hpp>
#include <miopen/common.hpp>

namespace miopen {

miopenStatus_t RoIAlignForward(Handle& handle,
                               const TensorDescriptor& inputDesc,
                               ConstData_t input,
                               const TensorDescriptor& roisDesc,
                               ConstData_t rois,
                               const TensorDescriptor& outputDesc,
                               Data_t output,
                               const int alignedHeight,
                               const int alignedWidth,
                               const float spatialScale,
                               const int samplingRatio,
                               const bool aligned,
                               const int roi_batch_index);

miopenStatus_t RoIAlignBackward(Handle& handle,
                                const TensorDescriptor& gradOutputDesc,
                                const void* gradOutput,
                                const TensorDescriptor& roisDesc,
                                const void* rois,
                                const TensorDescriptor& gradInputDesc,
                                void* gradInput,
                                const int alignedHeight,
                                const int alignedWidth,
                                const float spatialScale,
                                const int samplingRatio,
                                const bool aligned,
                                const int roi_batch_index);

} // namespace miopen
#endif // MIOPEN_ROI_ALIGN_HPP_
