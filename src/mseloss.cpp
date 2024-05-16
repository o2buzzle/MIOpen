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
                                    const void* x,
                                    const void* y,
                                    void* z,
                                    const float lossScale)
{
    const auto problem =
        miopen::mseloss::forward::ProblemDescription{xDesc, yDesc, zDesc, x, y, z, lossScale};

    const auto invoke_params = [&]() {
        auto tmp      = miopen::mseloss::forward::InvokeParams{};
        tmp.xDesc     = &xDesc;
        tmp.yDesc     = &yDesc;
        tmp.zDesc     = &zDesc;
        tmp.x         = x;
        tmp.y         = y;
        tmp.z         = z;
        tmp.lossScale = lossScale;
        return tmp;
    }();

    const auto algo = AlgorithmName{"MSELossForward"};
    const auto solvers =
        solver::SolverContainer<miopen::solver::mseloss::forward::MSELossForward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);
    return miopenStatusSuccess;
}

miopenStatus_t miopenMSELossBackward(Handle& handle,
                                     const TensorDescriptor& xDesc,
                                     const TensorDescriptor& yDesc,
                                     const TensorDescriptor& dzDesc,
                                     const TensorDescriptor& dxDesc,
                                     const TensorDescriptor& dyDesc,
                                     const void* x,
                                     const void* y,
                                     const void* dz,
                                     void* dx,
                                     void* dy,
                                     const float lossScale)
{

    const auto problem = mseloss::backward::ProblemDescription{
        xDesc, yDesc, dzDesc, dxDesc, dyDesc, x, y, dz, dx, dy, lossScale};

    const auto invoke_params = [&]() {
        auto tmp      = mseloss::backward::InvokeParams{};
        tmp.xDesc     = &xDesc;
        tmp.yDesc     = &yDesc;
        tmp.dzDesc    = &dzDesc;
        tmp.dxDesc    = &dxDesc;
        tmp.dyDesc    = &dyDesc;
        tmp.x         = x;
        tmp.y         = y;
        tmp.dz        = dz;
        tmp.dx        = dx;
        tmp.dy        = dy;
        tmp.lossScale = lossScale;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"MSELossBackward"};
    const auto solvers = solver::SolverContainer<solver::mseloss::backward::MSELossBackward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);
    return miopenStatusSuccess;
}
} // namespace miopen
