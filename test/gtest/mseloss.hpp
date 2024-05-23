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

#ifndef GUARD_CPU_MSE_LOSS_HPP
#define GUARD_CPU_MSE_LOSS_HPP

#include "get_handle.hpp"
#include "miopen/miopen.h"
#include "random.hpp"
#include "miopen/allocator.hpp"
#include "tensor_holder.hpp"
#include "miopen/mseloss.hpp"
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

struct MSELossTestCase
{
    size_t N;
    size_t C;
    size_t D;
    size_t H;
    size_t W;

    friend std::ostream& operator<<(std::ostream& os, const MSELossTestCase& tc)
    {
        return os << "(N: " << tc.N << " C:" << tc.C << " D:" << tc.D << " H:" << tc.H
                  << " W:" << tc.W << " )";
    }

    std::vector<size_t> GetInput()
    {
        if((N != 0) && (C != 0) && (D != 0) && (H != 0) && (W != 0))
        {
            return std::vector<size_t>({N, C, D, H, W});
        }
        else if((N != 0) && (C != 0) && (H != 0) && (W != 0))
        {
            return std::vector<size_t>({N, C, H, W});
        }
        else if((N != 0) && (C != 0) && (W != 0))
        {
            return std::vector<size_t>({N, C, W});
        }
        else if((N != 0) && (W != 0))
        {
            return std::vector<size_t>({N, W});
        }
        else if((N != 0))
        {
            return std::vector<size_t>({N});
        }
        else
        {
            std::cout << "Error Input Tensor Lengths\n" << std::endl;
            return std::vector<size_t>({0});
        }
    }
};

std::vector<MSELossTestCase> MSELossTestConfigs()
{
    return {{8, 512, 0, 0, 384},
            {8, 511, 0, 0, 1},
            {8, 512, 0, 0, 384},
            {16, 512, 0, 0, 384},
            {16, 512, 0, 0, 8},
            {48, 8, 0, 512, 512},
            {48, 8, 0, 512, 512},
            {16, 311, 1, 98, 512},
            {16, 311, 1, 98, 512}};
}

inline std::vector<size_t> GetStrides(std::vector<size_t> input, bool contiguous)
{
    if(!contiguous)
        std::swap(input.front(), input.back());
    std::vector<size_t> strides(input.size());
    strides.back() = 1;
    for(int i = input.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * input[i + 1];
    if(!contiguous)
        std::swap(strides.front(), strides.back());
    return strides;
}

template <typename T>
struct MSELossTest : public ::testing::TestWithParam<MSELossTestCase>
{
protected:
    MSELossTestCase mseloss_config;

    tensor<T> input;
    tensor<T> target;
    tensor<T> output;

    tensor<T> input_grad;
    tensor<T> target_grad;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;

    miopen::Allocator::ManageDataPtr input_grad_dev;
    miopen::Allocator::ManageDataPtr target_grad_dev;

    uint64_t divisor = 0;

    void SetUp() override
    {
        auto&& handle  = get_handle();
        mseloss_config = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        auto in_dims = mseloss_config.GetInput();
        auto strides = GetStrides(in_dims, false);

        input       = tensor<T>{in_dims}.generate(gen_value);
        target      = tensor<T>{in_dims}.generate(gen_value);
        input_grad  = tensor<T>{in_dims};
        target_grad = tensor<T>{in_dims};

        input_dev  = handle.Write(input.data);
        target_dev = handle.Write(target.data);

        if(divisor == 0)
        {
            output = tensor<T>{in_dims};
        }
        else
        {
            output = tensor<T>{{1}};
        }

        std::fill(input_grad.begin(), input_grad.end(), std::numeric_limits<T>::quiet_NaN());
        std::fill(target_grad.begin(), target_grad.end(), std::numeric_limits<T>::quiet_NaN());
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());

        input_grad_dev  = handle.Write(input_grad.data);
        target_grad_dev = handle.Write(target_grad.data);
        output_dev      = handle.Write(output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();
        auto outDesc  = output.desc;

        if(divisor != 0)
        {
            size_t workspace_in_bytes = 0;
            auto status               = miopenGetMSELossForwardWorkspaceSize(
                handle, input.desc, target.desc, &workspace_in_bytes);

            if(status != miopenStatusSuccess)
            {
                std::cout << "Error: failed to obtain workspace size" << std::endl;
            }

            workspace_dev = handle.Write(workspace_in_bytes);

            status = miopenMSELossForward(handle,
                                          input.desc,
                                          target.desc,
                                          output.desc,
                                          input_dev.get(),
                                          target_dev.get(),
                                          output_dev.get(),
                                          workspace_dev.get(),
                                          divisor);

            if(status != miopenStatusSuccess)
            {
                std::cout << "Error: failed to perform forward pass" << std::endl;
            }

            handle.Read(output.data, output_dev.get());
        }
        else
        {
            auto status = miopenMSELossForwardUnreduced(handle,
                                                        input.desc,
                                                        target.desc,
                                                        output.desc,
                                                        input_dev.get(),
                                                        target_dev.get(),
                                                        output_dev.get());

            if(status != miopenStatusSuccess)
            {
                std::cout << "Error: failed to perform forward pass" << std::endl;
            }

            handle.Read(output.data, output_dev.get());
        }
    }

    void Verify() { return true; }
}

#endif