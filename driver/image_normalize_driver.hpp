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

#ifndef GUARD_MIOPEN_IMAGE_NORMALIZE_DRIVER_HPP
#define GUARD_MIOPEN_IMAGE_NORMALIZE_DRIVER_HPP

#include "../test/tensor_holder.hpp"
#include "InputFlags.hpp"
#include "driver.hpp"
#include "image_adjust_driver_common.hpp"
#include "miopen/miopen.h"
#include "miopen/tensor.hpp"
#include "miopen/tensor_view.hpp"
#include "tensor_driver.hpp"
#include "tensor_view.hpp"
#include "timer.hpp"
#include <memory>

template <typename Tgpu, typename Tref>
void mloImageNormalizeRunHost(miopen::TensorDescriptor inputDesc,
                              miopen::TensorDescriptor outputDesc,
                              miopen::TensorDescriptor meanDesc,
                              miopen::TensorDescriptor stdvarDesc,
                              const Tgpu* input,
                              Tref* output,
                              const Tgpu* mean,
                              const Tgpu* stdvar)
{
    tensor_view_4d_t input_tv  = get_inner_expanded_4d_tv(inputDesc);
    tensor_view_4d_t output_tv = get_inner_expanded_4d_tv(outputDesc);
    tensor_view_4d_t mean_tv   = get_inner_expanded_4d_tv(meanDesc);
    tensor_view_4d_t stdvar_tv = get_inner_expanded_4d_tv(stdvarDesc);

    auto N         = inputDesc.GetElementSize();
    auto C         = input_tv.size[1];
    auto c_strides = input_tv.stride[1];

    for(size_t gid = 0; gid < N; gid++)
    {
        auto c = gid / c_strides % C;

        Tref pixel  = get4DValueAt(input, input_tv, gid);
        Tref result = (pixel - static_cast<Tref>(mean[c + mean_tv.offset])) /
                      static_cast<Tref>(stdvar[c + stdvar_tv.offset]);
        set4DValueAt(output, output_tv, gid, result);
    }
}

template <typename Tgpu, typename Tref>
class ImageNormalizeDriver : public Driver
{
public:
    ImageNormalizeDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputTensorDesc);
        miopenCreateTensorDescriptor(&outputTensorDesc);
        miopenCreateTensorDescriptor(&meanTensorDesc);
        miopenCreateTensorDescriptor(&stdvarTensorDesc);

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

    ~ImageNormalizeDriver() override
    {
        miopenDestroyTensorDescriptor(inputTensorDesc);
        miopenDestroyTensorDescriptor(outputTensorDesc);
        miopenDestroyTensorDescriptor(meanTensorDesc);
        miopenDestroyTensorDescriptor(stdvarTensorDesc);
    }

private:
    InputFlags inflags;
    int forw;

    miopenTensorDescriptor_t inputTensorDesc;
    miopenTensorDescriptor_t outputTensorDesc;
    miopenTensorDescriptor_t meanTensorDesc;
    miopenTensorDescriptor_t stdvarTensorDesc;

    std::unique_ptr<GPUMem> input_gpu;
    std::unique_ptr<GPUMem> output_gpu;
    std::unique_ptr<GPUMem> mean_gpu;
    std::unique_ptr<GPUMem> stdvar_gpu;

    std::vector<Tgpu> input_host;
    std::vector<Tgpu> output_host;
    std::vector<Tgpu> mean_host;
    std::vector<Tgpu> stdvar_host;

    std::vector<Tref> out_ref;
};

template <typename Tgpu, typename Tref>
int ImageNormalizeDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only the forward pass", "int");
    inflags.AddTensorFlag("input", 'I', "1x3x96x96", "Input Tensor Size");
    inflags.AddTensorFlag("mean", 'M', "3", "Mean Tensor Sequence Size");
    inflags.AddTensorFlag("stdvar", 'S', "3", "Stdvar Tensor Sequence Size");
    inflags.AddInputFlag("contiguous", 'Z', "1", "Use Contiguous Tensors", "int");

    inflags.AddInputFlag("iter", 'i', "10", "Number of iterations", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ImageNormalizeDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ImageNormalizeDriver<Tgpu, Tref>::GetandSetData()
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

    TensorParameters mean_vec   = inflags.GetValueTensor("mean");
    TensorParameters stdvar_vec = inflags.GetValueTensor("stdvar");

    // Assert that these are at least as large as the input channels
    assert(mean_vec.lengths[0] >= input_vec.lengths[1]);
    assert(stdvar_vec.lengths[0] >= input_vec.lengths[1]);

    SetTensorNd(meanTensorDesc, mean_vec.lengths, data_type);
    SetTensorNd(stdvarTensorDesc, stdvar_vec.lengths, data_type);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ImageNormalizeDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t in_size     = GetTensorSize(inputTensorDesc);
    size_t mean_size   = GetTensorSize(meanTensorDesc);
    size_t stdvar_size = GetTensorSize(stdvarTensorDesc);
    size_t out_size    = GetTensorSize(outputTensorDesc);

    uint32_t ctx = 0;

    input_gpu  = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_size, sizeof(Tgpu)));
    output_gpu = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_size, sizeof(Tgpu)));
    mean_gpu   = std::unique_ptr<GPUMem>(new GPUMem(ctx, mean_size, sizeof(Tgpu)));
    stdvar_gpu = std::unique_ptr<GPUMem>(new GPUMem(ctx, stdvar_size, sizeof(Tgpu)));

    input_host  = std::vector<Tgpu>(in_size, static_cast<Tgpu>(0));
    output_host = std::vector<Tgpu>(out_size, static_cast<Tgpu>(0));
    mean_host   = std::vector<Tgpu>(mean_size, static_cast<Tgpu>(0));
    stdvar_host = std::vector<Tgpu>(stdvar_size, static_cast<Tgpu>(0));
    out_ref     = std::vector<Tref>(out_size, static_cast<Tref>(0));

    for(auto i = 0; i < in_size; i++)
    {
        input_host[i] = static_cast<Tgpu>(prng::gen_A_to_B(0.0f, 1.0f));
    }

    for(auto i = 0; i < mean_size; i++)
    {
        mean_host[i] = static_cast<Tgpu>(prng::gen_A_to_B(0.0f, 1.0f));
    }

    for(auto i = 0; i < stdvar_size; i++)
    {
        stdvar_host[i] = static_cast<Tgpu>(prng::gen_A_to_B(0.0f, 1.0f));
    }

    if(input_gpu->ToGPU(GetStream(), input_host.data()) != miopenStatusSuccess)
        std::cerr << "Error copying buffer to GPU with size " << input_gpu->GetSize() << std::endl;

    if(mean_gpu->ToGPU(GetStream(), mean_host.data()) != miopenStatusSuccess)
        std::cerr << "Error copying buffer to GPU with size " << mean_gpu->GetSize() << std::endl;

    if(stdvar_gpu->ToGPU(GetStream(), stdvar_host.data()) != miopenStatusSuccess)
        std::cerr << "Error copying buffer to GPU with size " << stdvar_gpu->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ImageNormalizeDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME
    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenImageNormalize(GetHandle(),
                             inputTensorDesc,
                             meanTensorDesc,
                             stdvarTensorDesc,
                             outputTensorDesc,
                             input_gpu->GetMem(),
                             mean_gpu->GetMem(),
                             stdvar_gpu->GetMem(),
                             output_gpu->GetMem());

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
            std::cout << "ImageNormalize Forward Time Elapsed: " << kernel_total_time << " ms"
                      << std::endl;

        float kernel_avg_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_total_time;
        std::cout << "ImageNormalize Kernel Time Elapsed: " << kernel_avg_time << " ms"
                  << std::endl;
    }

    if(output_gpu->FromGPU(GetStream(), output_host.data()) != miopenStatusSuccess)
    {
        std::cerr << "Error copying buffer from GPU with size " << output_gpu->GetSize()
                  << std::endl;
        return -1;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ImageNormalizeDriver<Tgpu, Tref>::RunBackwardGPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ImageNormalizeDriver<Tgpu, Tref>::RunForwardCPU()
{
    mloImageNormalizeRunHost(miopen::deref(inputTensorDesc),
                             miopen::deref(outputTensorDesc),
                             miopen::deref(meanTensorDesc),
                             miopen::deref(stdvarTensorDesc),
                             input_host.data(),
                             out_ref.data(),
                             mean_host.data(),
                             stdvar_host.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ImageNormalizeDriver<Tgpu, Tref>::RunBackwardCPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ImageNormalizeDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();

    for(auto i = 0; i < out_ref.size(); i++)
    {
        if(out_ref[i] != output_host[i])
        {
            std::cerr << "out_ref[" << i << "] = " << out_ref[i] << " != output_host[" << i
                      << "] = " << output_host[i] << std::endl;
        }
        else
        {
            std::cout << "out_host[" << i << "]: " << out_ref[i] << std::endl;
        }
    }

    printf("Verification completed\n");
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ImageNormalizeDriver<Tgpu, Tref>::VerifyBackward()
{
    return miopenStatusSuccess;
}

#endif
