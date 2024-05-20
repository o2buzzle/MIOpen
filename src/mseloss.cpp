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

#include "miopen/common.hpp"
#include "miopen/execution_context.hpp"
#include "miopen/miopen.h"
#include "miopen/names.hpp"
#include "miopen/tensor.hpp"
#include "miopen/find_solution.hpp"
#include <miopen/mseloss.hpp>
#include <miopen/mseloss/problem_description.hpp>
#include <miopen/mseloss/invoke_params.hpp>
#include <miopen/mseloss/solvers.hpp>

namespace miopen {

miopenStatus_t miopenMSELossForward(Handle& handle,
                                    const TensorDescriptor& xDesc,
                                    const TensorDescriptor& yDesc,
                                    const TensorDescriptor& zDesc,
                                    ConstData_t x,
                                    ConstData_t y,
                                    Data_t z,
                                    float divisor)
{
    auto ctx = ExecutionContext{&handle};

    const auto problem = mseloss::forward::ProblemDescription{xDesc, yDesc, zDesc};

    const auto invoke_params = [&]() {
        auto tmp    = mseloss::forward::InvokeParams{};
        tmp.xDesc   = &xDesc;
        tmp.yDesc   = &yDesc;
        tmp.zDesc   = &zDesc;
        tmp.x       = x;
        tmp.y       = y;
        tmp.z       = z;
        tmp.divisor = divisor;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"MSELossForward"};
    const auto solvers = solver::SolverContainer<solver::mseloss::forward::MSELossForward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);
    return miopenStatusSuccess;
}

miopenStatus_t miopenMSELossForwardUnreduced(Handle& handle,
                                             const TensorDescriptor& xDesc,
                                             const TensorDescriptor& yDesc,
                                             const TensorDescriptor& zDesc,
                                             ConstData_t x,
                                             ConstData_t y,
                                             Data_t z)
{
    auto ctx = ExecutionContext{&handle};

    const auto problem = mseloss::forward_unreduced::ProblemDescription{xDesc, yDesc, zDesc};

    const auto invoke_params = [&]() {
        auto tmp  = mseloss::forward_unreduced::InvokeParams{};
        tmp.xDesc = &xDesc;
        tmp.yDesc = &yDesc;
        tmp.zDesc = &zDesc;
        tmp.x     = x;
        tmp.y     = y;
        tmp.z     = z;
        return tmp;
    }();

    const auto algo = AlgorithmName{"MSELossForwardUnreduced"};
    const auto solvers =
        solver::SolverContainer<solver::mseloss::forward_unreduced::MSELossForwardUnreduced>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t miopenMSELossBackward(Handle& handle,
                                     const TensorDescriptor& xDesc,
                                     const TensorDescriptor& yDesc,
                                     const TensorDescriptor& zDesc,
                                     const TensorDescriptor& dxDesc,
                                     const TensorDescriptor& dyDesc,
                                     ConstData_t x,
                                     ConstData_t y,
                                     ConstData_t z,
                                     Data_t dx,
                                     Data_t dy,
                                     float divisor)
{
    auto ctx = ExecutionContext{&handle};

    const auto problem = mseloss::backward::ProblemDescription{xDesc, yDesc, zDesc, dxDesc, dyDesc};

    const auto invoke_params = [&]() {
        auto tmp    = mseloss::backward::InvokeParams{};
        tmp.xDesc   = &xDesc;
        tmp.yDesc   = &yDesc;
        tmp.zDesc   = &zDesc;
        tmp.dxDesc  = &dxDesc;
        tmp.dyDesc  = &dyDesc;
        tmp.x       = x;
        tmp.y       = y;
        tmp.z       = z;
        tmp.dx      = dx;
        tmp.dy      = dy;
        tmp.divisor = divisor;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"MSELossBackward"};
    const auto solvers = solver::SolverContainer<solver::mseloss::backward::MSELossBackward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);
    return miopenStatusSuccess;
}

miopenStatus_t miopenMSELossBackwardUnreduced(Handle& handle,
                                              const TensorDescriptor& xDesc,
                                              const TensorDescriptor& yDesc,
                                              const TensorDescriptor& zDesc,
                                              const TensorDescriptor& dxDesc,
                                              const TensorDescriptor& dyDesc,
                                              ConstData_t x,
                                              ConstData_t y,
                                              ConstData_t z,
                                              Data_t dx,
                                              Data_t dy)
{
    auto ctx = ExecutionContext{&handle};

    const auto problem =
        mseloss::backward_unreduced::ProblemDescription{xDesc, yDesc, zDesc, dxDesc, dyDesc};

    const auto invoke_params = [&]() {
        auto tmp   = mseloss::backward_unreduced::InvokeParams{};
        tmp.xDesc  = &xDesc;
        tmp.yDesc  = &yDesc;
        tmp.zDesc  = &zDesc;
        tmp.dxDesc = &dxDesc;
        tmp.dyDesc = &dyDesc;
        tmp.x      = x;
        tmp.y      = y;
        tmp.z      = z;
        tmp.dx     = dx;
        tmp.dy     = dy;
        return tmp;
    }();

    const auto algo = AlgorithmName{"MSELossBackwardUnreduced"};
    const auto solvers =
        solver::SolverContainer<solver::mseloss::backward_unreduced::MSELossBackwardUnreduced>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);
    return miopenStatusSuccess;
}

} // namespace miopen
