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

#include "miopen/problem_description_base.hpp"
#include "miopen/tensor.hpp"
#include <cstdint>
namespace miopen {

struct NetworkConfig;

namespace roialign {
struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(const TensorDescriptor& inputDesc_,
                       const TensorDescriptor& roisDesc_,
                       const TensorDescriptor& outputDesc_,
                       const int alignedHeight_,
                       const int alignedWidth_)
        : inputDesc(inputDesc_),
          roisDesc(roisDesc_),
          outputDesc(outputDesc_),
          alignedHeight(alignedHeight_),
          alignedWidth(alignedWidth_)
    {
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetRoisDesc() const { return roisDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }

    int32_t GetAlignedHeight() const { return alignedHeight; }
    int32_t GetAlignedWidth() const { return alignedWidth; }

    NetworkConfig MakeNetworkConfig() const override;

private:
    const TensorDescriptor& inputDesc;
    const TensorDescriptor& roisDesc;
    const TensorDescriptor& outputDesc;

    const int32_t alignedHeight;
    const int32_t alignedWidth;
};
} // namespace roialign
} // namespace miopen
