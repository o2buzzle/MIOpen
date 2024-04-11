/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include "../driver/tensor_driver.hpp"
#include "get_handle.hpp"
#include "miopen/allocator.hpp"
#include "miopen/tensor.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/constant_pad.hpp>

struct PadConstantTestCase
{
    size_t N;
    size_t C;
    size_t D;
    size_t H;
    size_t W;
    int32_t dim;
    friend std::ostream& operator<<(std::ostream& os, const PadConstantTestCase& tc)
    {
        return os << " N:" << tc.N << " C:" << tc.C << " D:" << tc.D << " H:" << tc.H
                  << " W:" << tc.W << " dim:" << tc.dim;
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
        else if(N != 0)
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

std::vector<PadConstantTestCase> PadConstantTestConfigs()
{
    // clang-format off
    return {
        PadConstantTestCase{.N = 8, .C = 120, .D = 0, .H = 0, .W = 1, .dim = 0},
        PadConstantTestCase{.N = 8, .C = 120, .D = 0, .H = 0, .W = 1, .dim = 0},
        PadConstantTestCase{.N = 8, .C = 1023, .D = 0, .H = 0, .W = 1, .dim = 0},
    };
    // clang-format on
}

template <typename T = float>
struct PadConstantTest : public ::testing::TestWithParam<PadConstantTestCase>
{
protected:
    PadConstantTestCase pad_constant_config;
    tensor<T> input;
    tensor<T> output;

    tensor<T> ref_output;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;

    // We only do 2d tests (for now)
    const int padding[10] = {0, 0, 0, 0, 0, 0, 2, 0, 2, 0};

    void SetUp() override
    {
        auto&& handle       = get_handle();
        pad_constant_config = GetParam();
        auto gen_value      = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        auto in_dims = std::vector<size_t>({1, 1, 1,6, 6});
        input        = tensor<T>{in_dims}.generate(gen_value);

        std::vector<size_t> out_dims = {1, 1, 1, 10, 10};
        output                       = tensor<T>{out_dims}.generate(gen_value);

        input_dev  = handle.Write(input.data);
        output_dev = handle.Write(output.data);
    };

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        const int* pd;
        hipMalloc(&pd, 10 * sizeof(int));
        hipMemcpy((void*)pd, padding, 10 * sizeof(int), hipMemcpyHostToDevice);

        status = miopen::PadConstantForward(
            handle, input.desc, output.desc, input_dev.get(), output_dev.get(), pd, 3.5f);
        EXPECT_EQ(status, miopenStatusSuccess);
    }

    void Verify()
    {
        // ...Do nothing, right now at least
        EXPECT_TRUE(true);
    }

    // Nothing else yet. We are just hoping the test even **run**
};
