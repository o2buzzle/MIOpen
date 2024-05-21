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
#include "miopen/tensor.hpp"
#include "miopen/tensor_view.hpp"
#include "random.hpp"
#include "tensor_driver.hpp"
#include <memory>
#include <vector>
#include "../test/tensor_holder.hpp"
#include "tensor_view.hpp"

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
void mloMSELossForwardUnreducedRunHost(miopenTensorDescriptor_t inputDesc,
                                       miopenTensorDescriptor_t targetDesc,
                                       miopenTensorDescriptor_t outputDesc,
                                       Tgpu* input,
                                       Tgpu* target,
                                       Tref* output)

{
    tensor_view_5d_t I_tv = get_inner_expanded_tv(miopen::deref(inputDesc));
    tensor_view_5d_t T_tv = get_inner_expanded_tv(miopen::deref(targetDesc));
    tensor_view_5d_t O_tv = get_inner_expanded_tv(miopen::deref(outputDesc));

    int64_t gid = 0;

    while(true)
    {
        size_t n0123 = gid / I_tv.size[4], n4 = gid % I_tv.size[4];
        size_t n012 = n0123 / I_tv.size[3], n3 = n0123 % I_tv.size[3];
        size_t n01 = n012 / I_tv.size[2], n2 = n012 % I_tv.size[2];
        size_t n0 = n01 / I_tv.size[1], n1 = n01 % I_tv.size[1];

        if(!(n0 < I_tv.size[0]))
            break;

        size_t Iidx = get5DIndexAt<size_t>(I_tv, n0, n1, n2, n3, n4);
        size_t Tidx = get5DIndexAt<size_t>(T_tv, n0, n1, n2, n3, n4);
        size_t Oidx = get5DIndexAt<size_t>(O_tv, n0, n1, n2, n3, n4);

        output[Oidx] =
            static_cast<Tref>((input[Iidx] - target[Tidx]) * (input[Iidx] - target[Tidx]));
        ++gid;
    }
}

template <typename Tgpu, typename Tref>
void mloMSELossBackwardUnreducedRunHost(miopenTensorDescriptor_t inputDesc,
                                        miopenTensorDescriptor_t targetDesc,
                                        miopenTensorDescriptor_t outputDesc,
                                        miopenTensorDescriptor_t inputGradDesc,
                                        miopenTensorDescriptor_t targetGradDesc,
                                        Tgpu* input,
                                        Tgpu* target,
                                        Tgpu* output,
                                        Tref* input_grad,
                                        Tref* target_grad)
{
    tensor_view_5d_t I_tv  = get_inner_expanded_tv(miopen::deref(inputDesc));
    tensor_view_5d_t T_tv  = get_inner_expanded_tv(miopen::deref(targetDesc));
    tensor_view_5d_t O_tv  = get_inner_expanded_tv(miopen::deref(outputDesc));
    tensor_view_5d_t IG_tv = get_inner_expanded_tv(miopen::deref(inputGradDesc));
    tensor_view_5d_t TG_tv = get_inner_expanded_tv(miopen::deref(targetGradDesc));

    int64_t gid = 0;

    while(true)
    {
        size_t n0123 = gid / I_tv.size[4], n4 = gid % I_tv.size[4];
        size_t n012 = n0123 / I_tv.size[3], n3 = n0123 % I_tv.size[3];
        size_t n01 = n012 / I_tv.size[2], n2 = n012 % I_tv.size[2];
        size_t n0 = n01 / I_tv.size[1], n1 = n01 % I_tv.size[1];

        if(!(n0 < I_tv.size[0]))
            break;

        size_t Iidx = get5DIndexAt<size_t>(I_tv, n0, n1, n2, n3, n4);
        size_t Tidx = get5DIndexAt<size_t>(T_tv, n0, n1, n2, n3, n4);
        size_t Oidx = get5DIndexAt<size_t>(O_tv, n0, n1, n2, n3, n4);

        Tref grad = 2.0f * (input[Iidx] - target[Tidx]) * (output[Oidx]);

        if(input_grad != nullptr)
        {
            size_t IGidx      = get5DIndexAt<size_t>(IG_tv, n0, n1, n2, n3, n4);
            input_grad[IGidx] = grad;
        }

        if(target_grad != nullptr)
        {
            size_t TGidx       = get5DIndexAt<size_t>(TG_tv, n0, n1, n2, n3, n4);
            target_grad[TGidx] = -grad;
        }

        ++gid;
    }
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
    inflags.AddInputFlag("forw", 'F', "0", "Run only Forward Cat (Default=1)", "int");
    inflags.AddInputFlag("contiguous", 'Z', "0", "Use Contiguous Tensors", "int");
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
    SetTensorNd(inputGradDesc, in_lengths, data_type);
    SetTensorNd(targetGradDesc, in_lengths, data_type);

    // Output is basically (input - target).pow(2) sized when unreduced
    // And (input - target).pow(2).(mean|sum)() (ala. 1) sized when reduced
    if(inflags.GetValueStr("reduction") == "none")
    {
        auto out_lengths = in_lengths;
        SetTensorNd(outputDesc, out_lengths, data_type);
    }
    else
    {
        std::vector<int> out_lengths(1, 1);
        SetTensorNd(outputDesc, out_lengths, data_type);
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
    input_grad  = std::vector<Tgpu>(input_size, static_cast<Tgpu>(0));
    target_grad = std::vector<Tgpu>(input_size, static_cast<Tgpu>(0));

    // Host side buffers (for verification)
    output_host      = std::vector<Tref>(output_size, static_cast<Tref>(0));
    input_grad_host  = std::vector<Tref>(input_size, static_cast<Tref>(0));
    target_grad_host = std::vector<Tref>(input_size, static_cast<Tref>(0));

    // Fill input and target tensors
    for(size_t i = 0; i < input.size(); i++)
    {
        input[i]  = prng::gen_A_to_B(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        target[i] = prng::gen_A_to_B(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        input[i]  = static_cast<Tgpu>(1.0);
        target[i] = static_cast<Tgpu>(1.0);
    }

    memset(input_grad.data(), 1, input_grad.size() * sizeof(Tgpu));
    memset(target_grad.data(), 1, target_grad.size() * sizeof(Tgpu));

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
    if(inflags.GetValueStr("reduction") == "none")
    {
        mloMSELossForwardUnreducedRunHost(
            inputDesc, targetDesc, outputDesc, input.data(), target.data(), output_host.data());
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MSELossDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();

    for(size_t i = 0; i < output.size(); i++)
    {
        if(output[i] != output_host[i])
        {
            std::cerr << "Error: Forward CPU and GPU mismatch" << std::endl;
            std::cerr << "output[" << i << "] = " << output[i] << " != " << output_host[i]
                      << std::endl;
            return -1;
        }
    }

    printf("Success: Forward CPU and GPU match\n");
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MSELossDriver<Tgpu, Tref>::RunBackwardGPU()
{
    if(inflags.GetValueStr("reduction") == "none")
    {
        for(size_t i = 0; i < inflags.GetValueInt("iter"); i++)
        {
            auto status = miopenMSELossBackwardUnreduced(GetHandle(),
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
    }
    else
    {
        return miopenStatusSuccess;
    }

    if(input_grad_buf->FromGPU(GetStream(), input_grad.data()) != miopenStatusSuccess)
        std::cerr << "Error: Failed to copy input_grad from GPU, size " << input_grad.size()
                  << std::endl;
    if(target_grad_buf->FromGPU(GetStream(), target_grad.data()) != miopenStatusSuccess)
        std::cerr << "Error: Failed to copy target_grad from GPU, size " << target_grad.size()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MSELossDriver<Tgpu, Tref>::RunBackwardCPU()
{
    if(inflags.GetValueStr("reduction") == "none")
    {
        mloMSELossBackwardUnreducedRunHost(inputDesc,
                                           targetDesc,
                                           outputDesc,
                                           inputGradDesc,
                                           targetGradDesc,
                                           input.data(),
                                           target.data(),
                                           output.data(),
                                           input_grad_host.data(),
                                           target_grad_host.data());
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MSELossDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();

    for(size_t i = 0; i < input_grad.size(); i++)
    {
        if(input_grad[i] != input_grad_host[i])
        {
            std::cerr << "Error: Backward CPU and GPU mismatch" << std::endl;
            std::cerr << "input_grad[" << i << "] = " << input_grad[i]
                      << " != " << input_grad_host[i] << std::endl;
            return -1;
        }
    }

    for(size_t i = 0; i < target_grad.size(); i++)
    {
        if(target_grad[i] != target_grad_host[i])
        {
            std::cerr << "Error: Backward CPU and GPU mismatch" << std::endl;
            std::cerr << "target_grad[" << i << "] = " << target_grad[i]
                      << " != " << target_grad_host[i] << std::endl;

            return -1;
        }
    }

    printf("Success: Backward CPU and GPU match\n");
    return miopenStatusSuccess;
}

#endif
