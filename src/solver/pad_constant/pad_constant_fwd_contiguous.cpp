/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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

namespace miopen {
namespace solver {
namespace pad_constant_fwd_contiguous {
bool PadConstantFwdContiguous::IsApplicable(
    const ExecutionContext& /*context*/,
    const miopen::pad_constant_fwd_contiguous::ProblemDescription& problem) const
{
    if(!problem.IsSameType())
    {
        return false;
    }

    return true;
}

ConvSolution PadConstantFwdContiguous::GetSolution(
    const ExecutionContext& /*context*/,
    const miopen::pad_constant_fwd_contiguous::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};
    auto ydims  = problem.GetYDesc().GetLengths();

    auto input_dtype  = miopen::GetDataType(problem.GetXDesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetYDesc().GetType());

    // for xgridsize: 5d -> 1d
    size_t output_size = 1;
    for(int i = 0; i < 5; i++)
    {
        output_size *= ydims[i] == 0 ? 1 : ydims[i];
    }

    size_t xlocalsize = 1024;
    // AlignUp blows up, because output_size can be > int_max. Lovely.
    // size_t xgridsize  = AlignUp(output_size, xlocalsize);
    size_t xgridsize =
        (((static_cast<size_t>(output_size) + xlocalsize - 1) / xlocalsize) * xlocalsize);
    size_t ylocalsize = 1;
    size_t ygridsize  = 1;

    auto kernel = KernelInfo{};

    kernel.kernel_file = "MIOpenPadConstantFwd.cpp";
    kernel.kernel_name = "PadConstantFwdContiguous";

    // TODO: Actually understand how to use this properly
    const auto build_params = KernelBuildParameters{
        {"INPUT_TYPE", input_dtype == "half" ? "half" : "float"},
        {"OUTPUT_TYPE", output_dtype == "half" ? "half" : "float"},
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
            decltype(auto) params =
                invoke_params.CastTo<miopen::pad_constant_fwd_contiguous::InvokeParams>();

            auto xdims = params.xDesc->GetLengths();
            auto ydims = params.yDesc->GetLengths();

            // Copy x_dims to GPU (needed to get position in kernel later)
            const size_t* d_xdims;
            hipMallocManaged(&d_xdims, xdims.size() * sizeof(size_t));
            memcpy((void*)d_xdims, xdims.data(), xdims.size() * sizeof(size_t));

            const size_t* d_ydims;
            hipMallocManaged(&d_ydims, ydims.size() * sizeof(size_t));
            memcpy((void*)d_ydims, ydims.data(), ydims.size() * sizeof(size_t));

            // Calculate output size (again)
            size_t output_size = 1;
            for(unsigned long ydim : ydims)
                output_size *= ydim;

            kernel(params.x,
                   params.y,
                   d_xdims,
                   d_ydims,
                   params.padding,
                   output_size,
                   params.padding_value);
            hipFree((void*)d_xdims);
            hipFree((void*)d_ydims);
        };
    };

    return result;
}
} // namespace pad_constant_fwd_contiguous
} // namespace solver
} // namespace miopen
