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

#include <miopen/common.hpp>
#include <miopen/errors.hpp>
#include <miopen/logger.hpp>
#include <miopen/roialign.hpp>
#include <miopen/miopen.h>

extern "C" miopenStatus_t miopenRoIAlignForward(miopenHandle_t handle,
                                                const miopenTensorDescriptor_t inputDesc,
                                                const void* input,
                                                const miopenTensorDescriptor_t roisDesc,
                                                const void* rois,
                                                const miopenTensorDescriptor_t outputDesc,
                                                void* output,
                                                int alignedHeight,
                                                int alignedWidth,
                                                float spatialScale,
                                                int samplingRatio,
                                                bool aligned,
                                                int roi_batch_index)
{
    return miopen::try_([&] {
        miopen::RoIAlignForward(miopen::deref(handle),
                                miopen::deref(inputDesc),
                                DataCast(input),
                                miopen::deref(roisDesc),
                                DataCast(rois),
                                miopen::deref(outputDesc),
                                DataCast(output),
                                alignedHeight,
                                alignedWidth,
                                spatialScale,
                                samplingRatio,
                                aligned,
                                roi_batch_index);
    });
}
