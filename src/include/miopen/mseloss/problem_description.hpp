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

#pragma once

#include "miopen/common.hpp"
#include <miopen/problem_description_base.hpp>
#include <miopen/activ.hpp>
#include <miopen/tensor.hpp>

namespace miopen {
struct NetworkConfig;
namespace mseloss {
namespace forward {
struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(const TensorDescriptor& xDesc_,
                       const TensorDescriptor& yDesc_,
                       const TensorDescriptor& zDesc_)
        : xDesc(xDesc_), yDesc(yDesc_), zDesc(zDesc_){};

    NetworkConfig MakeNetworkConfig() const override;

    const TensorDescriptor& GetXDesc() const { return xDesc; }
    const TensorDescriptor& GetYDesc() const { return yDesc; }
    const TensorDescriptor& GetZDesc() const { return zDesc; }

private:
    const TensorDescriptor& xDesc;
    const TensorDescriptor& yDesc;
    const TensorDescriptor& zDesc;
};
} // namespace forward

namespace forward_unreduced {
struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(const TensorDescriptor& xDesc_,
                       const TensorDescriptor& yDesc_,
                       const TensorDescriptor& zDesc_)
        : xDesc(xDesc_), yDesc(yDesc_), zDesc(zDesc_){};

    NetworkConfig MakeNetworkConfig() const override;

    const TensorDescriptor& GetXDesc() const { return xDesc; }
    const TensorDescriptor& GetYDesc() const { return yDesc; }
    const TensorDescriptor& GetZDesc() const { return zDesc; }

private:
    const TensorDescriptor& xDesc;
    const TensorDescriptor& yDesc;
    const TensorDescriptor& zDesc;
};
} // namespace forward_unreduced

namespace backward {
struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(const TensorDescriptor& xDesc_,
                       const TensorDescriptor& yDesc_,
                       const TensorDescriptor& zDesc_,
                       const TensorDescriptor& dxDesc_,
                       const TensorDescriptor& dyDesc_,
                       const float divisor_ = 1.0f)
        : xDesc(xDesc_),
          yDesc(yDesc_),
          dzDesc(zDesc_),
          dxDesc(dxDesc_),
          dyDesc(dyDesc_),
          divisor(divisor_){};

    NetworkConfig MakeNetworkConfig() const override;

    const TensorDescriptor& GetXDesc() const { return xDesc; }
    const TensorDescriptor& GetYDesc() const { return yDesc; }
    const TensorDescriptor& GetZDesc() const { return dzDesc; }

    const TensorDescriptor& GetDXDesc() const { return dxDesc; }
    const TensorDescriptor& GetDYDesc() const { return dyDesc; }

    float GetDivisor() const { return divisor; }

private:
    const TensorDescriptor& xDesc;
    const TensorDescriptor& yDesc;
    const TensorDescriptor& dzDesc;
    const TensorDescriptor& dxDesc;
    const TensorDescriptor& dyDesc;
    const float divisor = 1.0f;
};
} // namespace backward

namespace backward_unreduced {
struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(const TensorDescriptor& xDesc_,
                       const TensorDescriptor& yDesc_,
                       const TensorDescriptor& zDesc_,
                       const TensorDescriptor& dxDesc_,
                       const TensorDescriptor& dyDesc_)
        : xDesc(xDesc_), yDesc(yDesc_), zDesc(zDesc_), dxDesc(dxDesc_), dyDesc(dyDesc_){};

    NetworkConfig MakeNetworkConfig() const override;

    const TensorDescriptor& GetXDesc() const { return xDesc; }
    const TensorDescriptor& GetYDesc() const { return yDesc; }
    const TensorDescriptor& GetZDesc() const { return zDesc; }

    const TensorDescriptor& GetDXDesc() const { return dxDesc; }
    const TensorDescriptor& GetDYDesc() const { return dyDesc; }

private:
    const TensorDescriptor& xDesc;
    const TensorDescriptor& yDesc;
    const TensorDescriptor& zDesc;
    const TensorDescriptor& dxDesc;
    const TensorDescriptor& dyDesc;
};
} // namespace backward_unreduced
} // namespace mseloss
} // namespace miopen
