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

#ifndef GUARD_MIOPEN_MSELOSS_DRIVER_HPP
#define GUARD_MIOPEN_MSELOSS_DRIVER_HPP

#include "driver.hpp"
#include "miopen/miopen.h"
#include "random.hpp"
#include "tensor_driver.hpp"
#include <memory>
#include <vector>
#include "../test/tensor_holder.hpp"

inline std::vector<std::string> split(const std::string& s, char delim)
{
    std::vector<std::string> elems;
    std::stringstream ss(s);
    std::string item;
    while(std::getline(ss, item, delim))
    {
        elems.push_back(item);
    }
    return elems;
}

template <typename T>
inline std::vector<T> ComputeStrides(std::vector<T> input, bool contiguous)
{
    if(!contiguous)
        std::swap(input.front(), input.back());
    std::vector<T> strides(input.size());
    strides.back() = 1;
    for(int i = input.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * input[i + 1];
    if(!contiguous)
        std::swap(strides.front(), strides.back());
    return strides;
}

template <typename Tgpu, typename Tref>
class MSELossDriver : public Driver
{

public:
    MSELossDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&targetDesc);
        miopenCreateTensorDescriptor(&outputDesc);
        miopenCreateTensorDescriptor(&inputGradDesc);
        miopenCreateTensorDescriptor(&targetGradDesc);

        data_type = miopen_type<Tgpu>{};
    }
    ~MSELossDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(targetDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(inputGradDesc);
        miopenDestroyTensorDescriptor(targetGradDesc);
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }
    int GetandSetData() override;
    void GetInputTensorLengthsFromCmdLine();
    int AllocateBuffersAndCopy() override;
    int RunForwardGPU() override;
    int RunForwardCPU();
    int RunBackwardGPU() override;
    int RunBackwardCPU();

    int VerifyBackward() override;
    int VerifyForward() override;

private:
    InputFlags inflags;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t targetDesc;
    miopenTensorDescriptor_t outputDesc;

    miopenTensorDescriptor_t inputGradDesc;
    miopenTensorDescriptor_t targetGradDesc;

    std::unique_ptr<GPUMem> input_buf;
    std::unique_ptr<GPUMem> target_buf;
    std::unique_ptr<GPUMem> output_buf;

    std::unique_ptr<GPUMem> input_grad_buf;
    std::unique_ptr<GPUMem> target_grad_buf;

    std::vector<Tgpu> input;
    std::vector<Tgpu> target;
    std::vector<Tgpu> output;

    std::vector<Tgpu> input_grad;
    std::vector<Tgpu> target_grad;

    std::vector<Tref> output_host;
    std::vector<Tref> input_grad_host;
    std::vector<Tref> target_grad_host;

    Tgpu divisor;
};

template <typename Tgpu, typename Tref>
int MSELossDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward Cat (Default=1)", "int");
    inflags.AddInputFlag("contiguous", 'Z', "1", "Use Contiguous Tensors", "int");
    inflags.AddTensorFlag("in_tensors", 'I', "1", "Input Tensors");
    inflags.AddInputFlag("divisor", 'D', "1", "Divisor", "float");
    inflags.AddInputFlag("reduction", 'r', "none", "Reduction", "string");

    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MSELossDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
void MSELossDriver<Tgpu, Tref>::GetInputTensorLengthsFromCmdLine()
{
    auto in_lengths     = inflags.GetValueTensor("in_tensors").lengths;
    auto target_lengths = inflags.GetValueTensor("in_tensors").lengths;

    auto makeContiguous = inflags.GetValueInt("contiguous") == 1;

    auto input_strides  = ComputeStrides(in_lengths, makeContiguous);
    auto target_strides = ComputeStrides(target_lengths, makeContiguous);

    SetTensorNd(inputDesc, in_lengths, input_strides, data_type);
    SetTensorNd(targetDesc, target_lengths, target_strides, data_type);

    // Output is basically (input - target).pow(2) sized when unreduced
    // And (input - target).pow(2).(mean|sum)() (ala. 1) sized when reduced
    if(inflags.GetValueStr("reduction") == "none")
    {
        auto out_lengths = in_lengths;
        SetTensorNd(outputDesc, out_lengths, data_type);
        SetTensorNd(inputGradDesc, out_lengths, data_type);
        SetTensorNd(targetGradDesc, out_lengths, data_type);
    }
    else
    {
        std::vector<int> out_lengths(1, 1);
        SetTensorNd(outputDesc, out_lengths, data_type);
        SetTensorNd(inputGradDesc, out_lengths, data_type);
        SetTensorNd(targetGradDesc, out_lengths, data_type);
    }
}

template <typename Tgpu, typename Tref>
int MSELossDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t input_size  = GetTensorSize(inputDesc);
    size_t target_size = GetTensorSize(targetDesc);
    // Output is basically (input - target).pow(2) sized when unreduced
    // And (input - target).pow(2).(mean|sum)() (ala. 1) sized when reduced
    size_t output_size = inflags.GetValueStr("reduction") == "none" ? input_size : 1;

    input_buf  = std::unique_ptr<GPUMem>(new GPUMem(0, input_size, sizeof(Tgpu)));
    target_buf = std::unique_ptr<GPUMem>(new GPUMem(0, target_size, sizeof(Tgpu)));
    output_buf = std::unique_ptr<GPUMem>(new GPUMem(0, output_size, sizeof(Tgpu)));

    input_grad_buf  = std::unique_ptr<GPUMem>(new GPUMem(0, input_size, sizeof(Tgpu)));
    target_grad_buf = std::unique_ptr<GPUMem>(new GPUMem(0, target_size, sizeof(Tgpu)));

    input       = std::vector<Tgpu>(input_size, static_cast<Tgpu>(0));
    target      = std::vector<Tgpu>(target_size, static_cast<Tgpu>(0));
    output      = std::vector<Tgpu>(output_size, static_cast<Tgpu>(0));
    input_grad  = std::vector<Tgpu>(output_size, static_cast<Tgpu>(0));
    target_grad = std::vector<Tgpu>(output_size, static_cast<Tgpu>(0));

    // Host side buffers (for verification)
    output_host      = std::vector<Tref>(output_size, static_cast<Tref>(0));
    input_grad_host  = std::vector<Tref>(output_size, static_cast<Tref>(0));
    target_grad_host = std::vector<Tref>(output_size, static_cast<Tref>(0));

    // Fill input and target tensors
    for(size_t i = 0; i < input.size(); i++)
    {
        input[i]  = prng::gen_A_to_B(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        target[i] = prng::gen_A_to_B(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }

    for(size_t i = 0; i < output.size(); i++)
    {
        input_grad[i]  = static_cast<Tgpu>(1.0);
        target_grad[i] = static_cast<Tgpu>(1.0);
    }

    if(input_buf->ToGPU(GetStream(), input.data()) != miopenStatusSuccess)
        std::cerr << "Error: Failed to copy input to GPU, size " << input.size() << std::endl;

    if(target_buf->ToGPU(GetStream(), target.data()) != miopenStatusSuccess)
        std::cerr << "Error: Failed to copy target to GPU, size " << target.size() << std::endl;

    if(input_grad_buf->ToGPU(GetStream(), input_grad.data()) != miopenStatusSuccess)
        std::cerr << "Error: Failed to copy input_grad to GPU, size " << input_grad.size()
                  << std::endl;

    if(target_grad_buf->ToGPU(GetStream(), target_grad.data()) != miopenStatusSuccess)
        std::cerr << "Error: Failed to copy target_grad to GPU, size " << target_grad.size()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MSELossDriver<Tgpu, Tref>::GetandSetData()
{
    divisor = inflags.GetValueDouble("divisor");
    GetInputTensorLengthsFromCmdLine();
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MSELossDriver<Tgpu, Tref>::RunForwardGPU()
{
    if(inflags.GetValueStr("reduction") == "none")
    {
        for(size_t i = 0; i < inflags.GetValueInt("iter"); i++)
        {
            auto status = miopenMSELossForwardUnreduced(GetHandle(),
                                                        inputDesc,
                                                        targetDesc,
                                                        outputDesc,
                                                        input_buf->GetMem(),
                                                        target_buf->GetMem(),
                                                        output_buf->GetMem());

            if(status != miopenStatusSuccess)
            {
                std::cerr << "Error: miopenMSELossForwardUnreduced failed" << std::endl;
                return status;
            }
        }

        if(output_buf->FromGPU(GetStream(), output.data()) != miopenStatusSuccess)
            std::cerr << "Error: Failed to copy output from GPU, size " << output.size()
                      << std::endl;
    }
    else
    {
        for(size_t i = 0; i < inflags.GetValueInt("iter"); i++)
            return miopenStatusSuccess;
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MSELossDriver<Tgpu, Tref>::RunForwardCPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MSELossDriver<Tgpu, Tref>::VerifyForward()
{
    for(size_t i = 0; i < output.size(); i++)
    {
        printf("output[%lu] = %f\n", i, output[i]);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MSELossDriver<Tgpu, Tref>::RunBackwardGPU()
{
    if(inflags.GetValueStr("reduction") == "none")
    {
        for(size_t i = 0; i < inflags.GetValueInt("iter"); i++)
        {
            auto status = miopenMSELossBackward(GetHandle(),
                                                inputDesc,
                                                targetDesc,
                                                outputDesc,
                                                inputGradDesc,
                                                targetGradDesc,
                                                input_buf->GetMem(),
                                                target_buf->GetMem(),
                                                output_buf->GetMem(),
                                                input_grad_buf->GetMem(),
                                                target_grad_buf->GetMem());

            if(status != miopenStatusSuccess)
            {
                std::cerr << "Error: miopenMSELossBackward failed" << std::endl;
                return status;
            }
        }

        if(input_grad_buf->FromGPU(GetStream(), input_grad.data()) != miopenStatusSuccess)
            std::cerr << "Error: Failed to copy input_grad from GPU, size " << input_grad.size()
                      << std::endl;
    }
    else
    {
        return miopenStatusSuccess;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MSELossDriver<Tgpu, Tref>::RunBackwardCPU()
{
    for(size_t i = 0; i < input_grad.size(); i++)
    {
        printf("input_grad[%lu] = %f\n", i, input_grad[i]);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MSELossDriver<Tgpu, Tref>::VerifyBackward()
{
    return miopenStatusSuccess;
}

#endif
