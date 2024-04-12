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
#include "miopen/execution_context.hpp"
#include "miopen/kernel_build_params.hpp"
#include "miopen/miopen.h"
#include "miopen/mlo_internal.hpp"
#include "miopen/pad_constant_fwd/solvers.hpp"
#include "miopen/pad_constant_fwd/invoke_params.hpp"

namespace miopen {
namespace solver {
namespace pad_constant_fwd_contiguous {
bool PadConstantFwdContiguous::IsApplicable(
    const ExecutionContext& context,
    const miopen::pad_constant_fwd_contiguous::ProblemDescription& problem) const
{
    return true;
}

ConvSolution PadConstantFwdContiguous::GetSolution(
    const ExecutionContext& context,
    const miopen::pad_constant_fwd_contiguous::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype = problem.GetXDesc().GetType();
    auto ydims = problem.GetYDesc().GetLengths();

    // AlignUp(output_size / block_size, block_size)
    size_t xlocalsize = 256;
    size_t xgridsize  = 1;

    for (int i = 0; i < 5; i++)
    {
        xgridsize *= ydims[i] == 0 ? 1 : ydims[i];
    }
    xgridsize = AlignUp(xgridsize, xlocalsize);

    size_t ylocalsize = 1;
    size_t ygridsize  = 1;

    printf("xlocalsize = %lu, xgridsize = %lu\n", xlocalsize, xgridsize);
    printf("ylocalsize = %lu, ygridsize = %lu\n", ylocalsize, ygridsize);

    auto kernel = KernelInfo{};

    kernel.kernel_file = "MIOpenPadConstantFwd.cpp";
    kernel.kernel_name = "PadConstantFwdContiguous";
    
    // technically not necessary.
    const auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int32_t>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int32_t>(dtype == miopenFloat)},
        {"MIOPEN_USE_FP64", static_cast<int32_t>(dtype == miopenDouble)},
        {"MIOPEN_USE_BFP16", static_cast<int32_t>(dtype == miopenBFloat16)},
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

            const size_t* d_xdims;
            const size_t* d_ydims;


            hipMalloc(&d_xdims, xdims.size() * sizeof(size_t));
            hipMemcpy((void*)d_xdims, xdims.data(), xdims.size() * sizeof(size_t), hipMemcpyHostToDevice);
            hipMalloc(&d_ydims, ydims.size() * sizeof(size_t));
            hipMemcpy((void*)d_ydims, ydims.data(), ydims.size() * sizeof(size_t), hipMemcpyHostToDevice);

            // printf("Kernel reporting x_dim as n=%lu c=%lu d=%lu h=%lu w=%lu\n", d_xdims[0], d_xdims[1], d_xdims[2], d_xdims[3], d_xdims[4]);
            // printf("Kernel reporting y_dim as n=%lu c=%lu d=%lu h=%lu w=%lu\n", d_ydims[0], d_ydims[1], d_ydims[2], d_ydims[3], d_ydims[4]);
            // Calculate output size
            size_t output_size = 1;
            for(unsigned long ydim : ydims)
                output_size *= ydim;

            kernel(params.x, params.y, d_xdims, d_ydims, params.padding, output_size, params.padding_value);

            hipFree((void*)d_xdims);
            hipFree((void*)d_ydims);
        };
    };

    return result;
}
} // namespace pad_constant_fwd_contiguous
} // namespace solver
} // namespace miopen
