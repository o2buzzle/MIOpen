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

#include "miopen/bfloat16.hpp"
#include "miopen/conv_solution.hpp"
#include "miopen/invoke_params.hpp"
#include "miopen/kernel_build_params.hpp"
#include "miopen/mlo_internal.hpp"
#include "miopen/mseloss/solvers.hpp"
#include "miopen/mseloss/invoke_params.hpp"
#include "miopen/tensor_view.hpp"
#include <cstddef>

#define LOCAL_SIZE_MSELOSS 256

namespace miopen {
namespace solver {
namespace mseloss {
namespace forward {
bool MSELossForward::IsApplicable(const ExecutionContext& context,
                                  const miopen::mseloss::forward::ProblemDescription& problem) const
{
    return true;
}

ConvSolution
MSELossForward::GetSolution(const ExecutionContext& context,
                            const miopen::mseloss::forward::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype = problem.GetXDesc().GetType();
    auto xdims = problem.GetXDesc().GetLengths();
    auto ydims = problem.GetYDesc().GetLengths();

    auto numel = problem.GetXDesc().GetElementSize();

    size_t xlocalsize = LOCAL_SIZE_MSELOSS;
    size_t xgridsize  = AlignUp(numel, xlocalsize);
    size_t ylocalsize = 1;
    size_t ygridsize  = 1;
    size_t zlocalsize = 1;
    size_t zgridsize  = 1;

    auto kernel        = KernelInfo{};
    kernel.kernel_file = "MIOpenMSELoss.cpp";
    kernel.kernel_name = "MSELossForward5d";

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
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::mseloss::forward::InvokeParams>();

            auto xdims = params.xDesc->GetLengths();
            auto ydims = params.yDesc->GetLengths();

            auto xstrides = params.xDesc->GetStrides();
            auto ystrides = params.yDesc->GetStrides();

            tensor_view_5d_t I_tv = get_inner_expanded_tv(*params.xDesc);
            tensor_view_5d_t T_tv = get_inner_expanded_tv(*params.yDesc);

            kernel(params.x, params.y, params.workspace, params.divisor, I_tv, T_tv);
        };
    };
    return result;
}

std::size_t
MSELossForward::GetWorkspaceSize(const ExecutionContext& context,
                                 const miopen::mseloss::forward::ProblemDescription& problem) const
{
    auto tensor_dtype = problem.GetXDesc().GetType();
    auto numel        = problem.GetXDesc().GetElementSize();
    auto elsize       = [&]() -> std::size_t {
        switch(tensor_dtype)
        {
        case miopenHalf: return sizeof(half_float::half);
        case miopenFloat: return sizeof(float);
        case miopenBFloat16: return sizeof(bfloat16);
        default: break;
        }
        MIOPEN_THROW("Unsupported tensor type");
    }();

    return AlignUp(numel, LOCAL_SIZE_MSELOSS) * elsize;
}

} // namespace forward

namespace forward_unreduced {
bool MSELossForwardUnreduced::IsApplicable(
    const ExecutionContext& context,
    const miopen::mseloss::forward_unreduced::ProblemDescription& problem) const
{
    return true;
}

ConvSolution MSELossForwardUnreduced::GetSolution(
    const ExecutionContext& context,
    const miopen::mseloss::forward_unreduced::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype = problem.GetXDesc().GetType();
    auto xdims = problem.GetXDesc().GetLengths();
    auto ydims = problem.GetYDesc().GetLengths();

    auto numel = problem.GetXDesc().GetElementSize();

    size_t xlocalsize = LOCAL_SIZE_MSELOSS;
    size_t xgridsize  = AlignUp(numel, xlocalsize);
    size_t ylocalsize = 1;
    size_t ygridsize  = 1;
    size_t zlocalsize = 1;
    size_t zgridsize  = 1;

    auto kernel        = KernelInfo{};
    kernel.kernel_file = "MIOpenMSELoss.cpp";
    kernel.kernel_name = "MSELossForwardUnreduced5d";

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
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params =
                raw_params.CastTo<miopen::mseloss::forward_unreduced::InvokeParams>();

            auto I_tv = get_inner_expanded_tv(*params.xDesc);
            auto T_tv = get_inner_expanded_tv(*params.yDesc);
            auto O_tv = get_inner_expanded_tv(*params.zDesc);

            kernel(params.x, params.y, params.z, I_tv, T_tv, O_tv);
        };
    };

    return result;
}
} // namespace forward_unreduced

namespace backward {
bool MSELossBackward::IsApplicable(
    const ExecutionContext& context,
    const miopen::mseloss::backward::ProblemDescription& problem) const
{
    return true;
}

ConvSolution
MSELossBackward::GetSolution(const ExecutionContext& context,
                             const miopen::mseloss::backward::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype = problem.GetXDesc().GetType();
    auto xdims = problem.GetXDesc().GetLengths();
    auto ydims = problem.GetYDesc().GetLengths();

    auto numel = problem.GetDXDesc().GetElementSize();

    size_t xlocalsize = LOCAL_SIZE_MSELOSS;
    size_t xgridsize  = AlignUp(numel, xlocalsize);
    size_t ylocalsize = 1;
    size_t ygridsize  = 1;
    size_t zlocalsize = 1;
    size_t zgridsize  = 1;

    auto kernel        = KernelInfo{};
    kernel.kernel_file = "MIOpenMSELoss.cpp";
    kernel.kernel_name = "MSELossBackward5d";

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
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::mseloss::backward::InvokeParams>();

            tensor_view_5d_t I_tv  = get_inner_expanded_tv(*params.xDesc);
            tensor_view_5d_t T_tv  = get_inner_expanded_tv(*params.yDesc);
            tensor_view_5d_t dO_tv = get_inner_expanded_tv(*params.zDesc);
            tensor_view_5d_t dI_tv = get_inner_expanded_tv(*params.dxDesc);
            tensor_view_5d_t dT_tv = get_inner_expanded_tv(*params.dyDesc);

            kernel(params.x,
                   params.y,
                   params.z,
                   params.dx,
                   params.dy,
                   params.divisor,
                   I_tv,
                   T_tv,
                   dO_tv,
                   dI_tv,
                   dT_tv);
        };
    };
    return result;
}
} // namespace backward

namespace backward_unreduced {
bool MSELossBackwardUnreduced::IsApplicable(
    const ExecutionContext& context,
    const miopen::mseloss::backward_unreduced::ProblemDescription& problem) const
{
    return true;
}

ConvSolution MSELossBackwardUnreduced::GetSolution(
    const ExecutionContext& context,
    const miopen::mseloss::backward_unreduced::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype = problem.GetXDesc().GetType();
    auto xdims = problem.GetXDesc().GetLengths();
    auto ydims = problem.GetYDesc().GetLengths();

    auto numel = problem.GetDXDesc().GetElementSize();

    size_t xlocalsize = LOCAL_SIZE_MSELOSS;
    size_t xgridsize  = AlignUp(numel, xlocalsize);
    size_t ylocalsize = 1;
    size_t ygridsize  = 1;
    size_t zlocalsize = 1;
    size_t zgridsize  = 1;

    auto kernel        = KernelInfo{};
    kernel.kernel_file = "MIOpenMSELoss.cpp";
    kernel.kernel_name = "MSELossBackwardUnreduced5d";

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
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params =
                raw_params.CastTo<miopen::mseloss::backward_unreduced::InvokeParams>();

            tensor_view_5d_t I_tv  = get_inner_expanded_tv(*params.xDesc);
            tensor_view_5d_t T_tv  = get_inner_expanded_tv(*params.yDesc);
            tensor_view_5d_t dO_tv = get_inner_expanded_tv(*params.zDesc);
            tensor_view_5d_t dI_tv = get_inner_expanded_tv(*params.dxDesc);
            tensor_view_5d_t dT_tv = get_inner_expanded_tv(*params.dyDesc);

            kernel(params.x,
                   params.y,
                   params.z,
                   params.dx,
                   params.dy,
                   I_tv,
                   T_tv,
                   dO_tv,
                   dI_tv,
                   dT_tv);
        };
    };
    return result;
}
} // namespace backward_unreduced
} // namespace mseloss
} // namespace solver
} // namespace miopen
