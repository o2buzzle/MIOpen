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

#include "miopen/conv_solution.hpp"
#include "miopen/datatype.hpp"
#include "miopen/image_transform/adjust_saturation/invoke_params.hpp"
#include "miopen/image_transform/solvers.hpp"
#include "miopen/tensor.hpp"
#include "miopen/tensor_view.hpp"
#include <miopen/solver.hpp>

namespace miopen {
namespace solver {
namespace image_transform {
namespace adjust_saturation {

bool isDtypeFP(miopenDataType_t dtype)
{
    return dtype == miopenFloat || dtype == miopenHalf || dtype == miopenBFloat16;
}

bool ImageAdjustSaturation::IsApplicable(
    const ExecutionContext& /*context*/,
    const miopen::image_transform::adjust_saturation::ProblemDescription& problem) const
{
    if(!problem.IsSameType())
        return false;
    if(!problem.IsImprovementOverROCm())
        return false;
    return true;
}

size_t ImageAdjustSaturation::GetWorkspaceSize(
    const ExecutionContext& /*context*/,
    const miopen::image_transform::adjust_saturation::ProblemDescription& problem) const
{
    auto dtype   = problem.GetInputTensorDesc().GetType();
    auto numel   = problem.GetInputTensorDesc().GetElementSize();
    auto el_size = get_data_size(dtype);

    auto ws_size = numel * el_size;
    return ws_size;
}

ConvSolution ImageAdjustSaturation::GetSolution(
    const ExecutionContext& /*context*/,
    const miopen::image_transform::adjust_saturation::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype = problem.GetInputTensorDesc().GetType();
    auto numel = problem.GetInputTensorDesc().GetElementSize();

    size_t xlocalsize = 256;
    size_t xgridsize  = AlignUp(numel / 3, xlocalsize);
    size_t ylocalsize = 1;
    size_t ygridsize  = 1;
    size_t zlocalsize = 1;
    size_t zgridsize  = 1;

    // first, RGB to Grayscale
    auto rgb_kernel        = KernelInfo{};
    rgb_kernel.kernel_file = "MIOpenImageBlend.cpp";
    // if(problem.GetInputTensorDesc().IsContiguous() &&
    // problem.GetOutputTensorDesc().IsContiguous())
    //     rgb_kernel.kernel_name = "RGBToGrayscaleContiguous";
    // else
    rgb_kernel.kernel_name = "RGBToGrayscale";

    const auto build_params =
        KernelBuildParameters{{"MIOPEN_USE_FP16", static_cast<int32_t>(dtype == miopenHalf)},
                              {"MIOPEN_USE_FP32", static_cast<int32_t>(dtype == miopenFloat)},
                              {"MIOPEN_USE_FP64", static_cast<int32_t>(dtype == miopenDouble)},
                              {"MIOPEN_USE_BFP16", static_cast<int32_t>(dtype == miopenBFloat16)}};

    rgb_kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

    rgb_kernel.l_wk.push_back(xlocalsize);
    rgb_kernel.l_wk.push_back(ylocalsize);
    rgb_kernel.l_wk.push_back(zlocalsize);

    rgb_kernel.g_wk.push_back(xgridsize);
    rgb_kernel.g_wk.push_back(ygridsize);
    rgb_kernel.g_wk.push_back(zgridsize);

    result.construction_params.push_back(rgb_kernel);

    auto blend_kernel        = KernelInfo{};
    blend_kernel.kernel_file = "MIOpenImageBlend.cpp";
    if(problem.GetInputTensorDesc().IsContiguous() && problem.GetOutputTensorDesc().IsContiguous())
        blend_kernel.kernel_name = "BlendContiguous";
    else
        blend_kernel.kernel_name = "Blend";
    xgridsize = AlignUp(numel, xlocalsize);

    blend_kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

    blend_kernel.l_wk.push_back(xlocalsize);
    blend_kernel.l_wk.push_back(ylocalsize);
    blend_kernel.l_wk.push_back(zlocalsize);

    blend_kernel.g_wk.push_back(xgridsize);
    blend_kernel.g_wk.push_back(ygridsize);
    blend_kernel.g_wk.push_back(zgridsize);

    result.construction_params.push_back(blend_kernel);

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle, const AnyInvokeParams& raw_params) {
            decltype(auto) params =
                raw_params.CastTo<miopen::image_transform::adjust_saturation::InvokeParams>();

            auto elapsed = 0.0f;

            auto xdesc = miopen::deref(params.inputTensorDesc);
            auto ydesc = miopen::deref(params.outputTensorDesc);
            // WS is a contiguous tensor with the same shape as the input
            auto ws_desc = TensorDescriptor{xdesc.GetType(), xdesc.GetLengths()};

            auto dtype = xdesc.GetType();

            auto x_tv  = get_inner_expanded_4d_tv(xdesc);
            auto ws_tv = get_inner_expanded_4d_tv(ws_desc);
            auto y_tv  = get_inner_expanded_4d_tv(ydesc);

            auto numel    = xdesc.GetElementSize();
            auto c_stride = x_tv.size[2] * x_tv.size[3];
            auto n_stride = c_stride * x_tv.size[1];

            float bound = isDtypeFP(dtype) ? 1.0f : 255.0f;

            auto kernel = handle.Run(kernels[0]); // RGB to Grayscale
            if(kernel.name == "RGBToGrayscale")
            {
                kernel(params.input_buf, params.workspace_buf, x_tv, ws_tv, numel / 3);
            }
            else
            {
                kernel(params.input_buf,
                       params.workspace_buf,
                       x_tv.offset,
                       ws_tv.offset,
                       c_stride,
                       numel / 3);
            }

            if(handle.IsProfilingEnabled())
            {
                elapsed = handle.GetKernelTime();
            }

            kernel = handle.Run(kernels[1]); // Blend

            if(kernel.name == "Blend")
            {
                kernel(params.input_buf,
                       params.workspace_buf,
                       params.output_buf,
                       x_tv,
                       ws_tv,
                       y_tv,
                       n_stride,
                       c_stride,
                       numel,
                       params.saturation_factor,
                       bound);
            }
            else
            {
                kernel(params.input_buf,
                       params.workspace_buf,
                       params.output_buf,
                       x_tv.offset,
                       ws_tv.offset,
                       y_tv.offset,
                       n_stride,
                       c_stride,
                       numel,
                       params.saturation_factor,
                       bound);
            }

            if(handle.IsProfilingEnabled())
            {
                elapsed += handle.GetKernelTime();
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };

    return result;
}
} // namespace adjust_saturation
} // namespace image_transform
} // namespace solver
} // namespace miopen
