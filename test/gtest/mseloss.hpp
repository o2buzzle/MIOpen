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

#include "get_handle.hpp"
#include "miopen/miopen.h"
#include "cpu_mseloss.hpp"
#include "random.hpp"
#include "miopen/allocator.hpp"
#include "tensor_holder.hpp"
#include "miopen/mseloss.hpp"
#include "verify.hpp"
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

struct MSELossTestCase
{
    std::vector<size_t> lengths;
    float divisor;
    bool isContiguous;

    friend std::ostream& operator<<(std::ostream& os, const MSELossTestCase& tc)
    {
        os << " lengths:";
        for(int i = 0; i < tc.lengths.size(); i++)
        {
            auto input = tc.lengths[i];
            if(i != 0)
                os << ",";
            os << input;
        }
        os << " divisor:" << tc.divisor << " contiguous:" << tc.isContiguous;
        return os;
    }
};

std::vector<MSELossTestCase> MSELossTestConfigs()
{
    // clang-format off
    return {{{1, 2,3}, 1.0f, false},
            {{8, 8,8}, 1.0f, false},
            {{16, 128,384}, 1.0f, false},
            {{1,2,3,4}, 1.0f, false},
            {{8, 8, 8, 8}, 1.0f, false},
            {{16, 32, 32, 32}, 1.0f, false},
            {{1,1,16,1024}, 1.0f, false},
            {{16, 16, 32, 32, 2}, 1.0f, false},
            {{16, 16, 32, 32, 256}, 1.0f, false}};
    // clang-format on
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

template <class T>
struct MSELossTest : public ::testing::TestWithParam<MSELossTestCase>
{
protected:
    MSELossTestCase mseloss_config;

    tensor<T> input;
    tensor<T> target;
    tensor<T> output;
    tensor<T> output_ref;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;

    float divisor;

    void SetUp() override
    {
        auto&& handle  = get_handle();
        mseloss_config = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        auto in_dims = mseloss_config.lengths;
        auto strides = GetStrides(in_dims, mseloss_config.isContiguous);

        input  = tensor<T>{in_dims}.generate(gen_value);
        target = tensor<T>{in_dims}.generate(gen_value);

        input_dev  = handle.Write(input.data);
        target_dev = handle.Write(target.data);

        divisor = mseloss_config.divisor;

        if(divisor == 0.0f)
        {
            output     = tensor<T>{in_dims};
            output_ref = tensor<T>{in_dims};
        }
        else
        {
            output     = tensor<T>{{1}};
            output_ref = tensor<T>{{1}};
        }
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());
        std::fill(output_ref.begin(), output_ref.end(), std::numeric_limits<T>::quiet_NaN());

        output_dev = handle.Write(output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();
        auto outDesc  = output.desc;

        // forward portion
        cpu_mseloss_unreduced<T>(input.desc,
                                 target.desc,
                                 output.desc,
                                 input.data.data(),
                                 target.data.data(),
                                 output_ref.data.data());

        auto status = miopenMSELossForwardUnreduced(handle,
                                                    input.desc,
                                                    target.desc,
                                                    output.desc,
                                                    input_dev.get(),
                                                    target_dev.get(),
                                                    output_dev.get());

        EXPECT_EQ(status, miopenStatusSuccess);

        output.data = handle.Read<T>(output_dev, output.data.size());
    }

    void Verify()
    {
        // Forward verification
        auto error = miopen::rms_range(output_ref, output);
        EXPECT_TRUE(miopen::range_distance(output_ref) == miopen::range_distance(output));
        EXPECT_TRUE(error == 0) << "Outputs do not match each other. Error:" << error;
    }
};
