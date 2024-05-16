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
#include "miopen/execution_context.hpp"
#include "miopen/kernel_build_params.hpp"
#include "miopen/miopen.h"
#include "miopen/mlo_internal.hpp"
#include "miopen/pad_constant/solvers.hpp"
#include "miopen/pad_constant/invoke_params.hpp"
#include "miopen/tensor_view_5d.hpp"

namespace miopen {
namespace solver {
namespace pad_constant_fwd {
bool PadConstantFwd::IsApplicable(const ExecutionContext& /*context*/,
                                  const miopen::pad_constant_fwd::ProblemDescription& problem) const
{
    if(!problem.IsSameType())
        return false;
    if(!problem.IsSameShape())
        return false;
    if(!problem.IsPaddingValid())
        return false;
    if(!problem.IsImprovementOverROCm())
        return false;

    return true;
}

ConvSolution
PadConstantFwd::GetSolution(const ExecutionContext& /*context*/,
                            const miopen::pad_constant_fwd::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto input_dtype  = miopen::GetDataType(problem.GetXDesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetYDesc().GetType());

    size_t output_size = problem.GetYDesc().GetElementSize();

    size_t xlocalsize = 256;
    size_t xgridsize  = AlignUpUL(output_size, xlocalsize);
    size_t ylocalsize = 1;
    size_t ygridsize  = 1;

    auto kernel = KernelInfo{};

    kernel.kernel_file = "MIOpenPadConstant.cpp";
    if(problem.IsContiguous())
        kernel.kernel_name = "PadConstantFwdContiguous";
    else
        kernel.kernel_name = "PadConstantFwd";

    const auto build_params = KernelBuildParameters{
        {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
        {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
        {"MIOPEN_USE_FP64", static_cast<int>(output_dtype == "double")},
        {"MIOPEN_USE_FP32", static_cast<int>(output_dtype == "float")},
        {"MIOPEN_USE_FP16", static_cast<int>(output_dtype == "half")},
        {"MIOPEN_USE_BFP16", static_cast<int>(output_dtype == "bfloat16")},
    };

    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

    kernel.l_wk.push_back(xlocalsize);
    kernel.l_wk.push_back(ylocalsize);

    kernel.g_wk.push_back(xgridsize);
    kernel.g_wk.push_back(ygridsize);

    result.construction_params.push_back(kernel);

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& invoke_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = invoke_params.CastTo<miopen::pad_constant_fwd::InvokeParams>();

            tensor_view_5d_t input_tv  = get_inner_expanded_tv(*params.xDesc);
            tensor_view_5d_t output_tv = get_inner_expanded_tv(*params.yDesc);

            padding_5d_t padding;
            memset(padding.val, 0, sizeof(padding.val));
            for(auto i = 0; i < params.padding_size; i++)
                padding.val[i] = params.padding[i];

            size_t output_size = params.yDesc->GetElementSize();

            kernel(params.x,
                   params.y,
                   input_tv,
                   output_tv,
                   padding,
                   output_size,
                   params.padding_value);
        };
    };

    return result;
}
} // namespace pad_constant_fwd

namespace pad_constant_bwd {

bool PadConstantBwd::IsApplicable(const ExecutionContext& /*context*/,
                                  const miopen::pad_constant_bwd::ProblemDescription& problem) const
{
    if(!problem.IsSameType())
        return false;
    if(!problem.IsSameShape())
        return false;
    if(!problem.IsPaddingValid())
        return false;
    if(!problem.IsImprovementOverROCm())
        return false;

    return true;
}

ConvSolution
PadConstantBwd::GetSolution(const ExecutionContext& /*context*/,
                            const miopen::pad_constant_bwd::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto input_dtype  = miopen::GetDataType(problem.GetXDesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetYDesc().GetType());

    size_t input_size = problem.GetXDesc().GetElementSize();

    size_t xlocalsize = 256;
    size_t xgridsize  = AlignUpUL(input_size, xlocalsize);
    size_t ylocalsize = 1;
    size_t ygridsize  = 1;

    auto kernel = KernelInfo{};

    kernel.kernel_file = "MIOpenPadConstant.cpp";
    if(problem.IsContiguous())
        kernel.kernel_name = "PadConstantBwdContiguous";
    else
        kernel.kernel_name = "PadConstantBwd";

    const auto build_params = KernelBuildParameters{
        {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
        {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
        {"MIOPEN_USE_FP64", static_cast<int>(output_dtype == "double")},
        {"MIOPEN_USE_FP32", static_cast<int>(output_dtype == "float")},
        {"MIOPEN_USE_FP16", static_cast<int>(output_dtype == "half")},
        {"MIOPEN_USE_BFP16", static_cast<int>(output_dtype == "bfloat16")},
    };

    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

    kernel.l_wk.push_back(xlocalsize);
    kernel.l_wk.push_back(ylocalsize);

    kernel.g_wk.push_back(xgridsize);
    kernel.g_wk.push_back(ygridsize);

    result.construction_params.push_back(kernel);

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& invoke_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = invoke_params.CastTo<miopen::pad_constant_bwd::InvokeParams>();

            tensor_view_5d_t input_tv  = get_inner_expanded_tv(*params.dxDesc);
            tensor_view_5d_t output_tv = get_inner_expanded_tv(*params.dyDesc);

            padding_5d_t padding;
            for(auto i = 0; i < 10; i++)
                padding.val[i] = params.padding[i];

            // Calculate output size (again)
            size_t input_size = params.dxDesc->GetElementSize();

            kernel(params.dx, params.dy, input_tv, output_tv, padding, input_size);
        };
    };

    return result;
}

} // namespace pad_constant_bwd
} // namespace solver
} // namespace miopen
