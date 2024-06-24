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

#include "miopen/image_transform.hpp"
#include "miopen/common.hpp"
#include "miopen/execution_context.hpp"
#include "miopen/find_solution.hpp"
#include "miopen/image_transform/adjust_brightness/invoke_params.hpp"
#include "miopen/image_transform/adjust_brightness/problem_description.hpp"
#include "miopen/image_transform/adjust_contrast/invoke_params.hpp"
#include "miopen/image_transform/adjust_contrast/problem_description.hpp"
#include "miopen/image_transform/adjust_hue/invoke_params.hpp"
#include "miopen/image_transform/adjust_hue/problem_description.hpp"
#include "miopen/image_transform/adjust_saturation/invoke_params.hpp"
#include "miopen/image_transform/adjust_saturation/problem_description.hpp"
#include "miopen/image_transform/normalize/invoke_params.hpp"
#include "miopen/image_transform/normalize/problem_description.hpp"
#include "miopen/image_transform/solvers.hpp"
#include "miopen/miopen.h"
#include "miopen/names.hpp"
#include "miopen/tensor.hpp"
#include <cstddef>

namespace miopen {

miopenStatus_t ImageAdjustHue(Handle& handle,
                              const TensorDescriptor& inputTensorDesc,
                              const TensorDescriptor& outputTensorDesc,
                              ConstData_t input_buf,
                              Data_t output_buf,
                              float hue)
{
    auto ctx = ExecutionContext{&handle};
    const auto problem =
        image_transform::adjust_hue::ProblemDescription{inputTensorDesc, outputTensorDesc, hue};

    const auto invoke_params = [&]() {
        auto tmp             = image_transform::adjust_hue::InvokeParams{};
        tmp.inputTensorDesc  = &inputTensorDesc;
        tmp.outputTensorDesc = &outputTensorDesc;
        tmp.input_buf        = input_buf;
        tmp.output_buf       = output_buf;
        tmp.hue              = hue;
        return tmp;
    }();

    const auto algo = AlgorithmName{"ImageAdjustHue"};
    const auto solver =
        solver::SolverContainer<solver::image_transform::adjust_hue::ImageAdjustHue>{};

    solver.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t ImageAdjustBrightness(Handle& handle,
                                     const TensorDescriptor& inputTensorDesc,
                                     const TensorDescriptor& outputTensorDesc,
                                     ConstData_t input_buf,
                                     Data_t output_buf,
                                     const float brightness_factor)
{
    auto ctx           = ExecutionContext{&handle};
    const auto problem = image_transform::adjust_brightness::ProblemDescription{
        inputTensorDesc, outputTensorDesc, brightness_factor};

    const auto invoke_params = [&]() {
        auto tmp              = image_transform::adjust_brightness::InvokeParams{};
        tmp.inputTensorDesc   = &inputTensorDesc;
        tmp.outputTensorDesc  = &outputTensorDesc;
        tmp.input_buf         = input_buf;
        tmp.output_buf        = output_buf;
        tmp.brightness_factor = brightness_factor;
        return tmp;
    }();

    const auto algo   = AlgorithmName{"ImageAdjustBrightness"};
    const auto solver = solver::SolverContainer<
        solver::image_transform::adjust_brightness::ImageAdjustBrightness>{};

    solver.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t ImageNormalize(Handle& handle,
                              const TensorDescriptor& inputTensorDesc,
                              const TensorDescriptor& meanTensorDesc,
                              const TensorDescriptor& stdTensorDesc,
                              const TensorDescriptor& outputTensorDesc,
                              ConstData_t input_buf,
                              ConstData_t mean_buf,
                              ConstData_t std_buf,
                              Data_t output_buf)
{
    auto ctx           = ExecutionContext{&handle};
    const auto problem = image_transform::normalize::ProblemDescription{
        inputTensorDesc, meanTensorDesc, stdTensorDesc, outputTensorDesc};

    const auto invoke_params = [&]() {
        auto tmp             = image_transform::normalize::InvokeParams{};
        tmp.inputTensorDesc  = &inputTensorDesc;
        tmp.meanTensorDesc   = &meanTensorDesc;
        tmp.stddevTensorDesc = &stdTensorDesc;
        tmp.outputTensorDesc = &outputTensorDesc;
        tmp.input_buf        = input_buf;
        tmp.mean_buf         = mean_buf;
        tmp.stddev_buf       = std_buf;
        tmp.output_buf       = output_buf;
        return tmp;
    }();

    const auto algo = AlgorithmName{"ImageNormalize"};
    const auto solver =
        solver::SolverContainer<solver::image_transform::normalize::ImageNormalize>{};

    solver.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t ImageAdjustContrast(Handle& handle,
                                   const TensorDescriptor& inputTensorDesc,
                                   const TensorDescriptor& outputTensorDesc,
                                   ConstData_t input_buf,
                                   Data_t workspace_buf,
                                   Data_t output_buf,
                                   float contrast_factor)
{
    auto ctx = ExecutionContext{&handle};
    const auto problem =
        image_transform::adjust_contrast::ProblemDescription{inputTensorDesc, outputTensorDesc};

    const auto invoke_params = [&]() {
        auto tmp             = image_transform::adjust_contrast::InvokeParams{};
        tmp.inputTensorDesc  = &inputTensorDesc;
        tmp.outputTensorDesc = &outputTensorDesc;
        tmp.input_buf        = input_buf;
        tmp.workspace_buf    = workspace_buf;
        tmp.output_buf       = output_buf;
        tmp.contrast_factor  = contrast_factor;
        return tmp;
    }();

    const auto algo = AlgorithmName{"ImageAdjustContrast"};
    const auto solver =
        solver::SolverContainer<solver::image_transform::adjust_contrast::ImageAdjustContrast>{};

    solver.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

size_t ImageAdjustContrastGetWorkspaceSize(Handle& handle,
                                           const TensorDescriptor& inputTensorDesc,
                                           const TensorDescriptor& outputTensorDesc)
{
    auto ctx = ExecutionContext{&handle};
    const auto problem =
        image_transform::adjust_contrast::ProblemDescription{inputTensorDesc, outputTensorDesc};

    const auto algo = AlgorithmName{"ImageAdjustContrast"};

    const auto solvers =
        solver::SolverContainer<solver::image_transform::adjust_contrast::ImageAdjustContrast>{};

    auto workspace_sizes = solvers.GetWorkspaceSizes(ctx, problem);

    return workspace_sizes.empty() ? static_cast<size_t>(0) : workspace_sizes.front().second;
}

miopenStatus_t ImageAdjustSaturation(Handle& handle,
                                     const TensorDescriptor& inputTensorDesc,
                                     const TensorDescriptor& outputTensorDesc,
                                     ConstData_t input_buf,
                                     Data_t workspace_buf,
                                     Data_t output_buf,
                                     float saturation_factor)
{
    auto ctx           = ExecutionContext{&handle};
    const auto problem = image_transform::adjust_saturation::ProblemDescription{
        inputTensorDesc, outputTensorDesc, saturation_factor};

    const auto invoke_params = [&]() {
        auto tmp              = image_transform::adjust_saturation::InvokeParams{};
        tmp.inputTensorDesc   = &inputTensorDesc;
        tmp.outputTensorDesc  = &outputTensorDesc;
        tmp.input_buf         = input_buf;
        tmp.workspace_buf     = workspace_buf;
        tmp.output_buf        = output_buf;
        tmp.saturation_factor = saturation_factor;
        return tmp;
    }();

    const auto algo = AlgorithmName{"ImageAdjustSaturation"};

    const auto solvers = solver::SolverContainer<
        solver::image_transform::adjust_saturation::ImageAdjustSaturation>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

size_t ImageAdjustSaturationGetWorkspaceSize(Handle& handle,
                                             const TensorDescriptor& inputTensorDesc,
                                             const TensorDescriptor& outputTensorDesc,
                                             float saturation_factor)
{
    auto ctx           = ExecutionContext{&handle};
    const auto problem = image_transform::adjust_saturation::ProblemDescription{
        inputTensorDesc, outputTensorDesc, saturation_factor};

    const auto algo = AlgorithmName{"ImageAdjustSaturation"};

    const auto solvers = solver::SolverContainer<
        solver::image_transform::adjust_saturation::ImageAdjustSaturation>{};

    auto workspace_sizes = solvers.GetWorkspaceSizes(ctx, problem);

    return workspace_sizes.empty() ? static_cast<size_t>(0) : workspace_sizes.front().second;
}
} // namespace miopen
