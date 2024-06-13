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

#include "miopen/miopen.h"
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>
#include <miopen/image_transform.hpp>

extern "C" miopenStatus_t miopenImageAdjustHue(miopenHandle_t handle,
                                               miopenTensorDescriptor_t inputTensorDesc,
                                               miopenTensorDescriptor_t outputTensorDesc,
                                               const void* input_buf,
                                               void* output_buf,
                                               float hue)
{
    MIOPEN_LOG_FUNCTION(handle, inputTensorDesc, outputTensorDesc, input_buf, output_buf, hue);

    return miopen::try_([&] {
        miopen::miopenImageAdjustHue(miopen::deref(handle),
                                     miopen::deref(inputTensorDesc),
                                     miopen::deref(outputTensorDesc),
                                     DataCast(input_buf),
                                     DataCast(output_buf),
                                     hue);
    });
}

extern "C" miopenStatus_t miopenImageAdjustBrightness(miopenHandle_t handle,
                                                      miopenTensorDescriptor_t inputTensorDesc,
                                                      miopenTensorDescriptor_t outputTensorDesc,
                                                      const void* input_buf,
                                                      void* output_buf,
                                                      float brightness_factor)
{
    MIOPEN_LOG_FUNCTION(
        handle, inputTensorDesc, outputTensorDesc, input_buf, output_buf, brightness_factor);

    return miopen::try_([&] {
        miopen::miopenImageAdjustBrightness(miopen::deref(handle),
                                            miopen::deref(inputTensorDesc),
                                            miopen::deref(outputTensorDesc),
                                            DataCast(input_buf),
                                            DataCast(output_buf),
                                            brightness_factor);
    });
}

extern "C" miopenStatus_t miopenImageNormalize(miopenHandle_t handle,
                                               miopenTensorDescriptor_t inputTensorDesc,
                                               miopenTensorDescriptor_t meanTensorDesc,
                                               miopenTensorDescriptor_t stdTensorDesc,
                                               miopenTensorDescriptor_t outputTensorDesc,
                                               const void* input_buf,
                                               const void* mean_buf,
                                               const void* std_buf,
                                               void* output_buf)
{
    MIOPEN_LOG_FUNCTION(handle,
                        inputTensorDesc,
                        meanTensorDesc,
                        stdTensorDesc,
                        outputTensorDesc,
                        input_buf,
                        mean_buf,
                        std_buf,
                        output_buf);

    return miopen::try_([&] {
        miopen::miopenImageNormalize(miopen::deref(handle),
                                     miopen::deref(inputTensorDesc),
                                     miopen::deref(meanTensorDesc),
                                     miopen::deref(stdTensorDesc),
                                     miopen::deref(outputTensorDesc),
                                     DataCast(input_buf),
                                     DataCast(mean_buf),
                                     DataCast(std_buf),
                                     DataCast(output_buf));
    });
}

extern "C" miopenStatus_t miopenImageAdjustContrast(miopenHandle_t handle,
                                                    miopenTensorDescriptor_t inputTensorDesc,
                                                    miopenTensorDescriptor_t outputTensorDesc,
                                                    const void* input_buf,
                                                    void* workspace_buf,
                                                    void* output_buf,
                                                    float contrast_factor)
{
    MIOPEN_LOG_FUNCTION(handle,
                        inputTensorDesc,
                        outputTensorDesc,
                        input_buf,
                        workspace_buf,
                        output_buf,
                        contrast_factor);

    return miopen::try_([&] {
        miopen::miopenImageAdjustContrast(miopen::deref(handle),
                                          miopen::deref(inputTensorDesc),
                                          miopen::deref(outputTensorDesc),
                                          DataCast(input_buf),
                                          DataCast(workspace_buf),
                                          DataCast(output_buf),
                                          contrast_factor);
    });
}

extern "C" miopenStatus_t
miopenImageAdjustConstrastGetWorkspaceSize(miopenHandle_t handle,
                                           miopenTensorDescriptor_t inputTensorDesc,
                                           miopenTensorDescriptor_t outputTensorDesc,
                                           size_t* workspace_size)
{
    MIOPEN_LOG_FUNCTION(handle, inputTensorDesc, outputTensorDesc, workspace_size);
    return miopen::try_([&] {
        miopen::deref(workspace_size) = miopen::miopenImageAdjustContrastGetWorkspaceSize(
            miopen::deref(handle), miopen::deref(inputTensorDesc), miopen::deref(outputTensorDesc));
    });
}

extern "C" miopenStatus_t miopenImageAdjustSaturation(miopenHandle_t handle,
                                                      miopenTensorDescriptor_t inputTensorDesc,
                                                      miopenTensorDescriptor_t outputTensorDesc,
                                                      const void* input_buf,
                                                      void* workspace_buf,
                                                      void* output_buf,
                                                      float saturation_factor)
{
    MIOPEN_LOG_FUNCTION(handle,
                        inputTensorDesc,
                        outputTensorDesc,
                        input_buf,
                        workspace_buf,
                        output_buf,
                        saturation_factor);

    return miopen::try_([&] {
        miopen::miopenImageAdjustSaturation(miopen::deref(handle),
                                            miopen::deref(inputTensorDesc),
                                            miopen::deref(outputTensorDesc),
                                            DataCast(input_buf),
                                            DataCast(workspace_buf),
                                            DataCast(output_buf),
                                            saturation_factor);
    });
}

extern "C" miopenStatus_t
miopenImageAdjustSaturationGetWorkspaceSize(miopenHandle_t handle,
                                            miopenTensorDescriptor_t inputTensorDesc,
                                            miopenTensorDescriptor_t outputTensorDesc,
                                            size_t* workspace_size)
{
    MIOPEN_LOG_FUNCTION(handle, inputTensorDesc, outputTensorDesc, workspace_size);

    return miopen::try_([&] {
        miopen::deref(workspace_size) = miopen::miopenImageAdjustSaturationGetWorkspaceSize(
            miopen::deref(handle), miopen::deref(inputTensorDesc), miopen::deref(outputTensorDesc));
    });
}
