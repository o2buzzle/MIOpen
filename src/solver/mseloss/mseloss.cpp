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

#include "miopen/mseloss/solvers.hpp"

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
    return ConvSolution();
}
} // namespace forward
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
    return ConvSolution();
}

} // namespace backward
} // namespace mseloss
} // namespace solver
} // namespace miopen