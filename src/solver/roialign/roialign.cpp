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
#include "miopen/execution_context.hpp"
#include "miopen/kernel_build_params.hpp"
#include "miopen/miopen.h"
#include "miopen/mlo_internal.hpp"
#include "miopen/roialign/invoke_params.hpp"
#include "miopen/roialign/problem_description.hpp"
#include "miopen/roialign/solvers.hpp"
#include "miopen/tensor_view_utils.hpp"

#define ROIALIGN_LOCAL_SIZE 256

namespace miopen {
namespace solver {
namespace roialign {
namespace forward {

bool IsImprovementOverROCm(const ExecutionContext context,
                           const miopen::roialign::forward::ProblemDescription& problem)
{
    return true;
}

bool RoIAlignForward::IsApplicable(
    const ExecutionContext& context,
    const miopen::roialign::forward::ProblemDescription& problem) const
{
    return true;
}

ConvSolution
RoIAlignForward::GetSolution(const ExecutionContext& context,
                             const miopen::roialign::forward::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype       = problem.GetInputDesc().GetType();
    auto input_dims  = problem.GetInputDesc().GetLengths();
    auto rois_dims   = problem.GetRoisDesc().GetLengths();
    auto output_dims = problem.GetOutputDesc().GetLengths();

    const size_t K = problem.GetRoisDesc().GetLengths()[0];
    const size_t C = problem.GetInputDesc().GetLengths()[1];
    const size_t g = AlignUp(K * C * problem.GetAlignedHeight() * problem.GetAlignedWidth(),
                             ROIALIGN_LOCAL_SIZE);

    const size_t xlocalsize = ROIALIGN_LOCAL_SIZE;
    const size_t ylocalsize = 1;
    const size_t zlocalsize = 1;

    const size_t xgridsize = g + 1;
    const size_t ygridsize = 1;
    const size_t zgridsize = 1;

    auto kernel        = KernelInfo{};
    kernel.kernel_file = "MIOpenRoIAlign.cpp";
    kernel.kernel_name = "RoIAlignForward";

    const auto build_params =
        KernelBuildParameters{{"MIOPEN_USE_FP16", static_cast<int32_t>(dtype == miopenHalf)},
                              {"MIOPEN_USE_FP32", static_cast<int32_t>(dtype == miopenFloat)},
                              {"MIOPEN_USE_FP64", static_cast<int32_t>(dtype == miopenDouble)},
                              {"MIOPEN_USE_BFP16", static_cast<int32_t>(dtype == miopenBFloat16)}};

    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

    kernel.l_wk.push_back(xlocalsize);
    kernel.l_wk.push_back(ylocalsize);
    kernel.l_wk.push_back(zlocalsize);

    kernel.g_wk.push_back(xgridsize);
    kernel.g_wk.push_back(ygridsize);
    kernel.g_wk.push_back(zgridsize);

    result.construction_params.push_back(kernel);

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params = raw_params.CastTo<miopen::roialign::forward::InvokeParams>();
            decltype(auto) kernel = handle_.Run(kernels[0]);

            decltype(auto) input_tv  = get_inner_expanded_tv<4>(*params.inputDesc);
            decltype(auto) rois_tv   = get_inner_expanded_tv<2>(*params.roisDesc);
            decltype(auto) output_tv = get_inner_expanded_tv<4>(*params.outputDesc);

            kernel(params.input,
                   params.rois,
                   params.output,
                   params.alignedHeight,
                   params.alignedWidth,
                   params.spatialScale,
                   params.samplingRatio,
                   params.aligned,
                   params.roi_batch_index,
                   input_tv,
                   rois_tv,
                   output_tv);
        };
    };

    return result;
}

} // namespace forward
namespace backward {

bool IsImprovementOverROCm(const ExecutionContext context,
                           const miopen::roialign::backward::ProblemDescription& problem)
{
    return true;
}

bool RoIAlignBackward::IsApplicable(
    const ExecutionContext& context,
    const miopen::roialign::backward::ProblemDescription& problem) const
{
    return true;
}

ConvSolution
RoIAlignBackward::GetSolution(const ExecutionContext& context,
                              const miopen::roialign::backward::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype = problem.GetGradOutputDesc().GetType();
    auto K     = problem.GetRoisDesc().GetLengths()[0];
    auto C     = problem.GetGradOutputDesc().GetLengths()[1];

    auto xlocalsize = ROIALIGN_LOCAL_SIZE;
    auto xgridsize =
        AlignUp(K * C * problem.GetAlignedHeight() * problem.GetAlignedWidth(), xlocalsize);
    auto ylocalsize = 1;
    auto ygridsize  = 1;
    auto zlocalsize = 1;
    auto zgridsize  = 1;

    auto kernel        = KernelInfo{};
    kernel.kernel_file = "MIOpenRoIAlign.cpp";
    kernel.kernel_name = "RoIAlignBackward";

    const auto build_params =
        KernelBuildParameters{{"MIOPEN_USE_FP16", static_cast<int32_t>(dtype == miopenHalf)},
                              {"MIOPEN_USE_FP32", static_cast<int32_t>(dtype == miopenFloat)},
                              {"MIOPEN_USE_FP64", static_cast<int32_t>(dtype == miopenDouble)},
                              {"MIOPEN_USE_BFP16", static_cast<int32_t>(dtype == miopenBFloat16)}};

    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

    kernel.l_wk.push_back(xlocalsize);
    kernel.l_wk.push_back(ylocalsize);
    kernel.l_wk.push_back(zlocalsize);

    kernel.g_wk.push_back(xgridsize);
    kernel.g_wk.push_back(ygridsize);
    kernel.g_wk.push_back(zgridsize);

    result.construction_params.push_back(kernel);

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params = raw_params.CastTo<miopen::roialign::backward::InvokeParams>();
            decltype(auto) kernel = handle_.Run(kernels[0]);

            decltype(auto) grad_output_tv = get_inner_expanded_tv<4>(*params.gradOutputDesc);
            decltype(auto) rois_tv        = get_inner_expanded_tv<2>(*params.roisDesc);
            decltype(auto) grad_input_tv  = get_inner_expanded_tv<4>(*params.gradInputDesc);

            size_t N = grad_input_tv.size[0];
            size_t C = grad_input_tv.size[1];
            size_t H = grad_input_tv.size[2];
            size_t W = grad_input_tv.size[3];
            size_t K = rois_tv.size[0];

            kernel(params.gradOutput,
                   params.rois,
                   params.gradInput,
                   N,
                   C,
                   H,
                   W,
                   K,
                   params.alignedHeight,
                   params.alignedWidth,
                   params.spatialScale,
                   params.samplingRatio,
                   params.aligned,
                   params.roi_batch_index,
                   grad_output_tv,
                   rois_tv,
                   grad_input_tv);
        };
    };

    return result;
}

} // namespace backward
} // namespace roialign
} // namespace solver
} // namespace miopen
