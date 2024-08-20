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
#include "miopen/roialign/invoke_params.hpp"
#include "miopen/roialign/problem_description.hpp"
#include "miopen/roialign/solvers.hpp"
#include "miopen/tensor_view_utils.hpp"

#define ROIALIGN_LOCAL_SIZE 256

namespace miopen {
namespace solver {
namespace roialign {

bool IsImprovementOverROCm(const ExecutionContext context,
                           const miopen::roialign::ProblemDescription& problem)
{
    return true;
}

bool RoIAlignForward::IsApplicable(const ExecutionContext& context,
                                   const miopen::roialign::ProblemDescription& problem) const
{
    return true;
}

ConvSolution RoIAlignForward::GetSolution(const ExecutionContext& context,
                                          const miopen::roialign::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype       = problem.GetInputDesc().GetType();
    auto input_dims  = problem.GetInputDesc().GetLengths();
    auto rois_dims   = problem.GetRoisDesc().GetLengths();
    auto output_dims = problem.GetOutputDesc().GetLengths();

    const size_t K = problem.GetRoisDesc().GetLengths()[0];
    const size_t C = problem.GetInputDesc().GetLengths()[1];
    const size_t g =
        K * C * problem.GetAlignedHeight() * problem.GetAlignedWidth() / ROIALIGN_LOCAL_SIZE;

    const size_t xlocalsize = g + 1;
    const size_t ylocalsize = 1;
    const size_t zlocalsize = 1;

    const size_t xgridsize = ROIALIGN_LOCAL_SIZE;
    const size_t ygridsize = 1;
    const size_t zgridsize = 1;

    auto kernel        = KernelInfo{};
    kernel.kernel_file = "MIOpenRoIAlign.cpp";
    kernel.kernel_name = "RoIAlignForward";

    const auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int32_t>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int32_t>(dtype == miopenFloat)},
        {"MIOPEN_USE_FP64", static_cast<int32_t>(dtype == miopenDouble)},
        {"MIOPEN_USE_BFP16", static_cast<int32_t>(dtype == miopenBFloat16)},
    };

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
            decltype(auto) params = raw_params.CastTo<miopen::roialign::InvokeParams>();
            decltype(auto) kernel = handle_.Run(kernels[0]);

            auto input_tv  = get_inner_expanded_tv<4>(*params.inputDesc);
            auto rois_tv   = get_inner_expanded_tv<2>(*params.roisDesc);
            auto output_tv = get_inner_expanded_tv<4>(*params.outputDesc);

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

} // namespace roialign
} // namespace solver
} // namespace miopen
