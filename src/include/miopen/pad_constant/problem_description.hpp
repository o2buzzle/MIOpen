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

#include "miopen/names.hpp"
#include "miopen/problem_description_base.hpp"
#include "miopen/tensor.hpp"
#include <cstddef>
#include <miopen/miopen.h>

namespace miopen {
namespace pad_constant_fwd_contiguous {
struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(const TensorDescriptor& xDesc_,
                       const TensorDescriptor& yDesc_,
                       const size_t* padding_)
        : xDesc(xDesc_), yDesc(yDesc_), padding(padding_)
    {
    }

    const TensorDescriptor& GetXDesc() const { return xDesc; }
    const TensorDescriptor& GetYDesc() const { return yDesc; }

    NetworkConfig MakeNetworkConfig() const override;

    bool IsSameType() const
    {
        if(xDesc.GetType() == yDesc.GetType())
            return true;
        return false;
    }

    bool IsSameShape() const
    {
        auto xSize = xDesc.GetSize();
        auto ySize = yDesc.GetSize();
        if(xSize == ySize)
            return true;
        return false;
    }

    bool checkContiguous(const TensorDescriptor& x) const
    {
        size_t s = 1;
        for(int i = x.GetSize() - 1; i >= 0; --i)
        {
            if(s != x.GetStrides()[i])
            {
                return false;
            }
            s *= x.GetLengths()[i];
        }
        return true;
    }

    bool IsContiguous() const { return checkContiguous(xDesc) && checkContiguous(yDesc); }

    bool IsImprovementOverROCm() const
    {
        if(IsContiguous())
            // No contiguous case is faster
            return false;
        else
            // Appears to be faster if n isn't padded
            return padding[0] == 0 && padding[1] == 0;
    }

    bool IsPaddingValid() const
    {
        std::vector<size_t> input_and_padding = std::vector<size_t>(yDesc.GetLengths().size());

        for(int i = 0; i < xDesc.GetLengths().size(); i++)
        {
            input_and_padding[i] = xDesc.GetLengths()[i] + padding[2 * i] + padding[2 * i + 1];
            if(input_and_padding[i] != yDesc.GetLengths()[i])
            {
                return false;
            }
        }

        return true;
    }

private:
    const TensorDescriptor& xDesc;
    const TensorDescriptor& yDesc;
    const size_t* padding;
};
} // namespace pad_constant_fwd_contiguous

namespace pad_constant_bwd {
struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(const TensorDescriptor& xDesc_,
                       const TensorDescriptor& yDesc_,
                       const size_t* padding_)
        : xDesc(xDesc_), yDesc(yDesc_), padding(padding_)
    {
    }

    const TensorDescriptor& GetXDesc() const { return xDesc; }
    const TensorDescriptor& GetYDesc() const { return yDesc; }

    NetworkConfig MakeNetworkConfig() const override;

    bool IsSameType() const
    {
        if(xDesc.GetType() == yDesc.GetType())
            return true;
        return false;
    }

    bool IsSameShape() const
    {
        auto xSize = xDesc.GetSize();
        auto ySize = yDesc.GetSize();
        if(xSize == ySize)
            return true;
        return false;
    }

    bool checkContiguous(const TensorDescriptor& x) const
    {
        size_t s = 1;
        for(int i = x.GetSize() - 1; i >= 0; --i)
        {
            if(s != x.GetStrides()[i])
            {
                return false;
            }
            s *= x.GetLengths()[i];
        }
        return true;
    }

    bool IsContiguous() const { return checkContiguous(xDesc) && checkContiguous(yDesc); }

    bool IsImprovementOverROCm() const
    {
        if(IsContiguous())
        {
            // Contiguous case
            // No winning over ROCm, only ~7% of tested cases get > 20% improvement
            return false;
        }
        else
        {
            // Non contiguous case
            // We win over ROCm unless the only padded dimension is n
            bool all_zero = true;
            for(int i = 0; i < 10; i++)
            {
                if(i == 9 || i == 8)
                {
                    if(all_zero && padding[i] != 0)
                        return false;
                }
                else
                {
                    if(padding[i] != 0)
                        all_zero = false;
                }
            }
            return true;
        }
    }

    bool IsPaddingValid() const
    {
        std::vector<size_t> input_and_padding = std::vector<size_t>(yDesc.GetLengths().size());

        for(int i = 0; i < xDesc.GetLengths().size(); i++)
        {
            input_and_padding[i] = yDesc.GetLengths()[i] - padding[2 * i] - padding[2 * i + 1];
            if(input_and_padding[i] != xDesc.GetLengths()[i])
            {
                return false;
            }
        }

        return true;
    }

private:
    const TensorDescriptor& xDesc;
    const TensorDescriptor& yDesc;
    const size_t* padding;
};
} // namespace pad_constant_bwd
} // namespace miopen
