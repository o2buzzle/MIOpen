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
#include "miopen/allocator.hpp"
#include "miopen/image_transform.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include "gtest/cpu_image_adjust.hpp"
#include <algorithm>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
struct ImageAdjustSaturationTestCase
{
    size_t N;
    size_t C;
    size_t H;
    size_t W;
    float saturation;

    friend std::ostream& operator<<(std::ostream& os, const ImageAdjustSaturationTestCase& tc)
    {
        return os << " N:" << tc.N << " C:" << tc.C << " H:" << tc.H << " W:" << tc.W
                  << " saturation:" << tc.saturation;
    }

    std::vector<size_t> GetInput()
    {
        if((N != 0) && (C != 0) && (H != 0) && (W != 0))
        {
            return std::vector<size_t>({N, C, H, W});
        }
        else if((C != 0) && (H != 0) && (W != 0))
        {
            return std::vector<size_t>({1, C, H, W});
        }
        else
        {
            return std::vector<size_t>({1, 1, 1, 1});
        }
    }
};

inline std::vector<ImageAdjustSaturationTestCase> ImageAdjustSaturationTestConfigs()
{
    // clang-format off
    return {
        {1, 3, 96, 96, 0.2},
        {1, 3, 224, 224, 0.2},
        {1, 3, 300, 300, 0.2},
        {1, 3, 128, 128, 0.2},
        {1, 3, 500, 500, 0.2},
        {1, 3, 1024, 1024, 0.2},
        {8, 3, 96, 96, 0.2},
        {8, 3, 224, 224, 0.2},
        {8, 3, 300, 300, 0.2},
        {8, 3, 128, 128, 0.2},
        {8, 3, 500, 500, 0.2},
        {8, 3, 1024, 1024, 0.2},
        {16, 3, 96, 96, 0.2},
        {16, 3, 224, 224, 0.2},
        {16, 3, 300, 300, 0.2},
        {16, 3, 128, 128, 0.2},
        {16, 3, 500, 500, 0.2},
        {16, 3, 1024, 1024, 0.2}
    };
    // clang-format on
}

template <class T>
class ImageAdjustSaturationTest : public ::testing::TestWithParam<ImageAdjustSaturationTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle  = get_handle();
        test_config    = GetParam();
        auto gen_value = [](auto...) { return prng::gen_0_to_B(1.0f); };

        auto in_dims = test_config.GetInput();
        input        = tensor<T>{in_dims}.generate(gen_value);

        std::vector<size_t> out_dims(in_dims);

        output = tensor<T>{out_dims};
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());

        ref_output = tensor<T>{out_dims};
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());

        input_ptr  = handle.Write(input.data);
        output_ptr = handle.Write(output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();
        cpu_image_adjust_saturation(input, ref_output, test_config.saturation);

        workspace_size = miopen::ImageAdjustSaturationGetWorkspaceSize(
            handle, input.desc, output.desc, test_config.saturation);

        workspace_ptr = handle.Create(workspace_size);

        auto status = miopen::ImageAdjustSaturation(handle,
                                                    input.desc,
                                                    output.desc,
                                                    input_ptr.get(),
                                                    workspace_ptr.get(),
                                                    output_ptr.get(),
                                                    test_config.saturation);

        EXPECT_EQ(status, miopenStatusSuccess);

        output.data = handle.Read<T>(output_ptr, output.data.size());
    }

    void Verify()
    {
        auto threashold = std::numeric_limits<T>::epsilon();
        auto error      = miopen::rms_range(ref_output, output);

        EXPECT_TRUE(miopen::range_distance(ref_output) == miopen::range_distance(output));
        EXPECT_TRUE(error < threashold) << "Outputs do not match each other. Error:" << error;
    }

    ImageAdjustSaturationTestCase test_config;

    tensor<T> input;
    tensor<T> output;
    tensor<T> ref_output;

    size_t workspace_size = 0;

    miopen::Allocator::ManageDataPtr input_ptr;
    miopen::Allocator::ManageDataPtr output_ptr;
    miopen::Allocator::ManageDataPtr workspace_ptr;
};
