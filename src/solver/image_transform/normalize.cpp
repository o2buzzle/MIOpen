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

#include "miopen/image_transform/normalize/invoke_params.hpp"
#include "miopen/image_transform/solvers.hpp"
#include "miopen/kernel_build_params.hpp"
#include "miopen/kernel_info.hpp"
#include "miopen/miopen.h"
#include "miopen/tensor_view.hpp"
#include <cstddef>
namespace miopen {
namespace solver {
namespace image_transform {
namespace normalize {

bool ImageNormalize::IsApplicable(
    const ExecutionContext& context,
    const miopen::image_transform::normalize::ProblemDescription& problem) const
{
    if(!problem.IsSameType())
        return false;
    if(!problem.IsImprovementOverROCm())
        return false;
    if(!problem.IsInputSizesValid())
        return false;
    return true;
};

ConvSolution ImageNormalize::GetSolution(
    const ExecutionContext& context,
    const miopen::image_transform::normalize::ProblemDescription& problem) const
{

    auto result = ConvSolution{miopenStatusSuccess};
    auto dtype  = problem.GetInputTensorDesc().GetType();
    auto numel  = problem.GetInputTensorDesc().GetElementSize();

    size_t xlocalsize = 256;
    size_t xgridsize  = AlignUp(numel, xlocalsize);
    size_t ylocalsize = 1;
    size_t ygridsize  = 1;
    size_t zlocalsize = 1;
    size_t zgridsize  = 1;

    auto kernel        = KernelInfo{};
    kernel.kernel_file = "MIOpenImageNormalize.cpp";
    if(problem.GetInputTensorDesc().IsContiguous() && problem.GetOutputTensorDesc().IsContiguous())
        kernel.kernel_name = "ImageNormalizeContiguous";
    else
        kernel.kernel_name = "ImageNormalize";

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
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params =
                raw_params.CastTo<miopen::image_transform::normalize::InvokeParams>();

            auto input_tv  = get_inner_expanded_4d_tv(miopen::deref(params.inputTensorDesc));
            auto output_tv = get_inner_expanded_4d_tv(miopen::deref(params.outputTensorDesc));
            auto mean_tv   = get_inner_expanded_4d_tv(miopen::deref(params.meanTensorDesc));
            auto std_tv    = get_inner_expanded_4d_tv(miopen::deref(params.stddevTensorDesc));

            size_t N        = miopen::deref(params.inputTensorDesc).GetElementSize();
            size_t C        = input_tv.size[1];
            size_t c_stride = input_tv.stride[1];

            if(kernel.name == "ImageNormalizeContiguous")
            {
                kernel(params.input_buf,
                       params.mean_buf,
                       params.stddev_buf,
                       params.output_buf,
                       input_tv.offset,
                       mean_tv.offset,
                       std_tv.offset,
                       output_tv.offset,
                       c_stride,
                       C,
                       N);
            }
            else
            {
                kernel(params.input_buf,
                       params.mean_buf,
                       params.stddev_buf,
                       params.output_buf,
                       input_tv,
                       mean_tv.offset,
                       std_tv.offset,
                       output_tv,
                       c_stride,
                       C,
                       N);
            }
        };
    };

    return result;
}
} // namespace normalize
} // namespace image_transform
} // namespace solver
} // namespace miopen
