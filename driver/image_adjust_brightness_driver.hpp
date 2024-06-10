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

#ifndef GUARD_MIOPEN_IMAGE_ADJUST_BRIGHTNESS_DRIVER_HPP
#define GUARD_MIOPEN_IMAGE_ADJUST_BRIGHTNESS_DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include "image_adjust_driver_common.hpp"
#include "miopen/miopen.h"
#include "../test/tensor_holder.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>

template <typename Tgpu, typename Tref>
class ImageAdjustBrightnessDriver : public Driver
{
public:
    ImageAdjustBrightnessDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputTensorDesc);
        miopenCreateTensorDescriptor(&outputTensorDesc);

        data_type = miopen_type<Tgpu>{};
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    int VerifyForward() override;
    int VerifyBackward() override;

    ~ImageAdjustBrightnessDriver() override
    {
        miopenDestroyTensorDescriptor(inputTensorDesc);
        miopenDestroyTensorDescriptor(outputTensorDesc);
    }

private:
    InputFlags inflags;

    int forw;

    miopenTensorDescriptor_t inputTensorDesc;
    miopenTensorDescriptor_t outputTensorDesc;

    std::unique_ptr<GPUMem> input_gpu;
    std::unique_ptr<GPUMem> output_gpu;

    std::vector<Tgpu> in_host;
    std::vector<Tgpu> out_host;

    std::vector<Tref> out_ref;

    float brightness_factor;
};

template <typename Tgpu, typename Tref>
int ImageAdjustBrightnessDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only the forward pass", "int");

    inflags.AddTensorFlag("input", 'I', "1x3x96x96", "Input Tensor Size");
    inflags.AddInputFlag("contiguous", 'Z', "1", "Use Contiguous Tensors", "int");
    inflags.AddInputFlag("brightness", 'B', "0.2", "Brightness factor", "double");

    inflags.AddInputFlag("iter", 'i', "10", "Number of iterations", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ImageAdjustBrightnessDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    brightness_factor = inflags.GetValueDouble("brightness");

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ImageAdjustBrightnessDriver<Tgpu, Tref>::GetandSetData()
{
    TensorParameters input_vec = inflags.GetValueTensor("input");
    assert(input_vec.lengths.size() == 4 || input_vec.lengths.size() == 3);
    if(input_vec.lengths.size() == 3)
    {
        // n=1
        input_vec.lengths.insert(input_vec.lengths.begin(), 1);
    }

    auto strides = ComputeStrides(input_vec.lengths, inflags.GetValueInt("contiguous") == 1);

    SetTensorNd(inputTensorDesc, input_vec.lengths, strides, data_type);
    SetTensorNd(outputTensorDesc, input_vec.lengths, strides, data_type);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ImageAdjustBrightnessDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t input_size  = GetTensorSize(inputTensorDesc);
    size_t output_size = GetTensorSize(outputTensorDesc);

    uint32_t ctx = 0;

    input_gpu  = std::unique_ptr<GPUMem>(new GPUMem(ctx, input_size, sizeof(Tgpu)));
    output_gpu = std::unique_ptr<GPUMem>(new GPUMem(ctx, output_size, sizeof(Tgpu)));

    in_host  = std::vector<Tgpu>(input_size);
    out_host = std::vector<Tgpu>(output_size);
    out_ref  = std::vector<Tref>(output_size);

    for(auto i = 0; i < input_size; i++)
    {
        in_host[i] = static_cast<Tgpu>(prng::gen_0_to_B(255) / 256.0f);
    }

    if(input_gpu->ToGPU(GetStream(), in_host.data()) != miopenStatusSuccess)
        std::cerr << "Error copying buffer to GPU with size " << input_gpu->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ImageAdjustBrightnessDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenImageAdjustBrightness(GetHandle(),
                                    inputTensorDesc,
                                    outputTensorDesc,
                                    input_gpu->GetMem(),
                                    output_gpu->GetMem(),
                                    brightness_factor);

        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);
        kernel_total_time += time;
        if(i == 0)
            kernel_first_time = time;
    }

    if(inflags.GetValueInt("time") == 1)
    {
        STOP_TIME
        int iter = inflags.GetValueInt("iter");

        if(WALL_CLOCK)
            std::cout << "Image Adjust Brightness Time Elapsed: " << kernel_total_time << " ms"
                      << std::endl;

        float kernel_avg_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_total_time;

        std::cout << "Image Adjust Brightness Kernel Time Elapsed: " << kernel_avg_time << " ms"
                  << std::endl;
    }

    if(output_gpu->FromGPU(GetStream(), out_host.data()) != miopenStatusSuccess)
    {
        std::cerr << "Error copying buffer from GPU with size " << output_gpu->GetSize()
                  << std::endl;
        return -1;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ImageAdjustBrightnessDriver<Tgpu, Tref>::RunForwardCPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ImageAdjustBrightnessDriver<Tgpu, Tref>::VerifyForward()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ImageAdjustBrightnessDriver<Tgpu, Tref>::RunBackwardGPU()
{
    // Does not exist
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ImageAdjustBrightnessDriver<Tgpu, Tref>::RunBackwardCPU()
{
    // Does not exist
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ImageAdjustBrightnessDriver<Tgpu, Tref>::VerifyBackward()
{
    // Does not exist
    return miopenStatusSuccess;
}

#endif
