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

#ifndef GUARD_MIOPEN_PAD_CONSTANT_DRIVER_HPP
#define GUARD_MIOPEN_PAD_CONSTANT_DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include "tensor_driver.hpp"
#include "random.hpp"
#include "timer.hpp"
#include <cstdio>
#include <vector>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <../test/verify.hpp>

template <typename T>
void inline getNCDHW(T* ncdhw, const T idx, const T size[5])
{
    ulong ncdh = (idx) / size[4];
    ncdhw[4]   = (idx) % size[4];
    ulong ncd  = ncdh / size[3];
    ncdhw[3]   = ncdh % size[3];
    ulong nc   = ncd / size[2];
    ncdhw[2]   = ncd % size[2];
    ncdhw[1]   = nc % size[1];
    ncdhw[0]   = nc / size[1];
}

template <typename T>
inline T
get5DValueAt(const T* x, const size_t* x_strides, size_t n, size_t c, size_t d, size_t h, size_t w)
{
    return x[n * x_strides[0] + c * x_strides[1] + d * x_strides[2] + h * x_strides[3] +
             w * x_strides[4]];
}

template <typename T, typename U>
inline void set5DValueAt(T* x, const size_t* x_sizes, const size_t* x_strides, size_t idx, U val)
{
    uint64_t o[5];
    o[4] = x_strides[0] * (size_t)((idx) / x_sizes[4] / x_sizes[3] / x_sizes[2] / x_sizes[1]);
    o[3] = x_strides[1] * ((size_t)((idx) / x_sizes[4] / x_sizes[3] / x_sizes[2]) % x_sizes[1]);
    o[2] = x_strides[2] * ((size_t)((idx) / x_sizes[4] / x_sizes[3]) % x_sizes[2]);
    o[1] = x_strides[3] * ((size_t)((idx) / x_sizes[4]) % x_sizes[3]);
    o[0] = x_strides[4] * ((idx) % x_sizes[4]);
    x[o[0] + o[1] + o[2] + o[3] + o[4]] = val;
}

template <typename Tgpu, typename Tcheck>
void mloConstantPadForwardRunHost(miopenTensorDescriptor_t inputDesc,
                                  miopenTensorDescriptor_t outputDesc,
                                  Tgpu* input,
                                  Tcheck* output_host,
                                  const size_t* padding,
                                  float value)
{
    size_t o[5];
    auto input_dims  = miopen::deref(inputDesc).GetLengths();
    auto output_dims = miopen::deref(outputDesc).GetLengths();

    auto input_strides  = miopen::deref(inputDesc).GetStrides();
    auto output_strides = miopen::deref(outputDesc).GetStrides();

    size_t output_size = miopen::deref(outputDesc).GetElementSize();

    for(size_t gid = 0; gid < output_size; ++gid)
    {
        bool flag = true;
        getNCDHW(o, gid, output_dims.data());

        for(int i = 0; i < 5; i++)
        {
            o[i] = o[i] - padding[2 * i];
            flag *= (o[i] < input_dims[i]);
        }

        auto val =
            flag ? get5DValueAt<Tgpu>(input, input_strides.data(), o[0], o[1], o[2], o[3], o[4])
                 : value;
        set5DValueAt(output_host, output_dims.data(), output_strides.data(), gid, val);
    }
}

template <typename Tgpu, typename Tcheck>
void mloConstantPadBackwardRunHost(miopenTensorDescriptor_t backwardOutputDesc,
                                   miopenTensorDescriptor_t inputGradDesc,
                                   Tcheck* backward_output,
                                   Tgpu* input_grad,
                                   const size_t* padding)
{
    size_t o[5];

    auto backward_output_dims    = miopen::deref(backwardOutputDesc).GetLengths();
    auto backward_output_strides = miopen::deref(backwardOutputDesc).GetStrides();

    auto input_grad_dims    = miopen::deref(inputGradDesc).GetLengths();
    auto input_grad_strides = miopen::deref(inputGradDesc).GetStrides();

    size_t backward_output_size = miopen::deref(backwardOutputDesc).GetElementSize();
    for(size_t gid = 0; gid < backward_output_size; ++gid)
    {
        bool flag = true;
        getNCDHW(o, gid, backward_output_dims.data());

        for(int i = 0; i < 5; i++)
        {
            o[i] = o[i] + padding[2 * i];
            flag *= (o[i] < input_grad_dims[i]);
        }

        if(flag)
        {
            auto val = get5DValueAt<Tcheck>(
                input_grad, input_grad_strides.data(), o[0], o[1], o[2], o[3], o[4]);
            set5DValueAt(backward_output,
                         backward_output_dims.data(),
                         backward_output_strides.data(),
                         gid,
                         val);
        }
    }
}

template <typename T>
inline std::vector<T> GetStrides(std::vector<T> input, bool contiguous)
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
class ConstantPadDriver : public Driver
{

public:
    ConstantPadDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&outputDesc);
        miopenCreateTensorDescriptor(&backwardOutputDesc);

        data_type = miopen_type<Tgpu>{};
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;
    std::vector<int> GetInputTensorLengthsFromCmdLine();
    std::vector<size_t> GetPaddingsFromCmdLine();

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    Tref GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~ConstantPadDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(backwardOutputDesc);
    }

private:
    InputFlags inflags;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t outputDesc;
    miopenTensorDescriptor_t backwardOutputDesc;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> out_dev;
    std::unique_ptr<GPUMem> backward_out_dev;

    std::vector<Tgpu> input;
    std::vector<Tgpu> output;
    std::vector<Tgpu> backward_output;
    std::vector<Tref> output_host;
    std::vector<Tgpu> backward_output_host;

    std::vector<size_t> padding;
    Tgpu value;
};

template <typename Tgpu, typename Tref>
int32_t ConstantPadDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int32_t ConstantPadDriver<Tgpu, Tref>::GetandSetData()
{
    padding                = GetPaddingsFromCmdLine();
    bool makeNonContiguous = inflags.GetValueInt("contiguous") == 0;

    std::vector<int> input_dims = GetInputTensorLengthsFromCmdLine();
    auto strides                = GetStrides(input_dims, makeNonContiguous);

    SetTensorNd(inputDesc, input_dims, strides, data_type);
    SetTensorNd(backwardOutputDesc, input_dims, strides, data_type);

    std::vector<int> output_dims;
    output_dims.reserve(input_dims.size());

    for(int i = 0; i < input_dims.size(); i++)
    {
        output_dims.push_back(input_dims[i] + 2 * padding[2 * i]);
    }

    SetTensorNd(outputDesc, output_dims, data_type);

    value = inflags.GetValueDouble("value");

    return 0;
}

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

template <typename Tgpu, typename Tref>
int ConstantPadDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "0", "Only run the Forward Pass (Default=1)", "int");
    inflags.AddInputFlag("contiguous", 'Z', "1", "Use contiguous tensor", "int");
    inflags.AddInputFlag("in", 'I', "3,3,3,3,3", "Input batch size (n,c,d,h,w)", "int");
    inflags.AddInputFlag("pad", 'P', "1,1,1,1,1", "Padding batch size (n,c,d,h,w)", "int");
    inflags.AddInputFlag("value", 'v', "0", "Padding value", "string");

    inflags.AddInputFlag("iter", 'i', "10", "Number of iterations", "int");
    inflags.AddInputFlag("verify", 'V', "0", "Verify results", "int");
    inflags.AddInputFlag("time", 't', "0", "Enable/Disable time measurement", "int");
    inflags.AddInputFlag("wall", 'l', "0", "Enable/Disable walltime measurement", "int");

    return 0;
}

template <typename Tgpu, typename Tref>
std::vector<int> ConstantPadDriver<Tgpu, Tref>::GetInputTensorLengthsFromCmdLine()
{
    auto in_str = inflags.GetValueStr("in");

    // Parse the comma_separated string
    int in_n = 0, in_c = 0, in_d = 0, in_h = 0, in_w = 0;
    std::vector<std::string> in = split(in_str, ',');

    assert(in.size() == 5);

    in_n = std::stoi(in[0]);
    in_c = std::stoi(in[1]);
    in_d = std::stoi(in[2]);
    in_h = std::stoi(in[3]);
    in_w = std::stoi(in[4]);

    if((in_n != 0) && (in_c != 0) && (in_d != 0) && (in_h != 0) && (in_w != 0))
    {
        return std::vector<int>({in_n, in_c, in_d, in_h, in_w});
    }
    else if((in_n != 0) && (in_c != 0) && (in_h != 0) && (in_w != 0))
    {
        return std::vector<int>({in_n, in_c, 1, in_h, in_w});
    }
    else if((in_n != 0) && (in_c != 0) && (in_w != 0))
    {
        return std::vector<int>({in_n, in_c, 1, 1, in_w});
    }
    else if((in_n != 0) && (in_w != 0))
    {
        return std::vector<int>({in_n, 1, 1, 1, in_w});
    }
    else if(in_n != 0)
    {
        return std::vector<int>({in_n, 1, 1, 1, 1});
    }
    else
    {
        std::cerr << "Error Input Tensor Lengths\n" << std::endl;
        return std::vector<int>({0});
    }
}

template <typename Tgpu, typename Tref>
std::vector<size_t> ConstantPadDriver<Tgpu, Tref>::GetPaddingsFromCmdLine()
{
    std::vector<size_t> paddings = std::vector<size_t>(10);

    auto pad                   = inflags.GetValueStr("pad");
    std::vector<std::string> p = split(pad, ',');

    assert(p.size() == 5);

    // Reversed to be consistent with how PyTorch pads (it pads last dimensions first)
    paddings[0] = std::stoi(p[4]);
    paddings[2] = std::stoi(p[3]);
    paddings[4] = std::stoi(p[2]);
    paddings[6] = std::stoi(p[1]);
    paddings[8] = std::stoi(p[0]);

    return paddings;
}

template <typename Tgpu, typename Tref>
int ConstantPadDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t input_size  = GetTensorSize(inputDesc);
    size_t output_size = GetTensorSize(outputDesc);

    in_dev           = std::unique_ptr<GPUMem>(new GPUMem(0, input_size, sizeof(Tgpu)));
    out_dev          = std::unique_ptr<GPUMem>(new GPUMem(0, output_size, sizeof(Tgpu)));
    backward_out_dev = std::unique_ptr<GPUMem>(new GPUMem(0, input_size, sizeof(Tgpu)));

    input       = std::vector<Tgpu>(input_size, static_cast<Tgpu>(0));
    output      = std::vector<Tgpu>(output_size, static_cast<Tgpu>(0));
    output_host = std::vector<Tref>(output_size, static_cast<Tref>(0));

    backward_output      = std::vector<Tgpu>(input_size, static_cast<Tgpu>(0));
    backward_output_host = std::vector<Tref>(input_size, static_cast<Tref>(0));

    for(int i = 0; i < input_size; i++)
    {
        input[i] = prng::gen_A_to_B(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }

    if(in_dev->ToGPU(GetStream(), input.data()) != 0)
        std::cerr << "ConstantPadDriver: Error copying data to GPU, size: " << in_dev->GetSize()
                  << std::endl;

    return 0;
}

template <typename Tgpu, typename Tref>
int ConstantPadDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenPadConstantFwd(GetHandle(),
                             inputDesc,
                             outputDesc,
                             in_dev->GetMem(),
                             out_dev->GetMem(),
                             padding.data(),
                             value);

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
            std::cout << "Wall-clock Time Elapsed: " << t.gettime_ms() / iter << " ms" << std::endl;

        float kernel_avg_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "Kernel Time Elapsed: " << kernel_avg_time << " ms" << std::endl;
    }

    if(out_dev->FromGPU(GetStream(), output.data()) != 0)
        std::cerr << "Error copying data from GPU, size: " << out_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ConstantPadDriver<Tgpu, Tref>::RunForwardCPU()
{
    mloConstantPadForwardRunHost(
        inputDesc, outputDesc, input.data(), output_host.data(), padding.data(), value);
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ConstantPadDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();

    bool failed = false;

    for(int i = 0; i < output.size(); i++)
    {
        if(output[i] != output_host[i])
        {
            std::cout << "output[" << i << "] = " << output[i] << " != "
                      << "output_host[" << i << "] = " << output_host[i] << std::endl;
            return -1;
        }
    }

    if(failed)
    {
        std::cout << "ConstantPadDriver: Forward verification failed." << std::endl;
        return -1;
    }

    std::cout << "ConstantPadDriver: Forward verification passed." << std::endl;
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ConstantPadDriver<Tgpu, Tref>::RunBackwardCPU()
{
    mloConstantPadBackwardRunHost(
        backwardOutputDesc, outputDesc, backward_output_host.data(), output.data(), padding.data());
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ConstantPadDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenPadConstantBwd(GetHandle(),
                             backwardOutputDesc,
                             outputDesc,
                             backward_out_dev->GetMem(),
                             out_dev->GetMem(),
                             padding.data());

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
            std::cout << "Wall-clock Backward Time Elapsed: " << t.gettime_ms() / iter << " ms"
                      << std::endl;

        float kernel_avg_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "Kernel Backward Time Elapsed: " << kernel_avg_time << " ms" << std::endl;
    }

    if(backward_out_dev->FromGPU(GetStream(), backward_output.data()) != 0)
        std::cerr << "Error copying data from GPU, size: " << in_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ConstantPadDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();

    bool failed = false;
    for(int i = 0; i < backward_output.size(); i++)
    {
        if(backward_output[i] != backward_output_host[i])
        {
            std::cout << "backward_output[" << i << "] = " << backward_output[i] << " != "
                      << "backward_output_host[" << i << "] = " << backward_output_host[i];
            return -1;
        }
    }

    if(failed)
    {
        std::cout << "ConstantPadDriver: Backward verification failed." << std::endl;
        return -1;
    }

    std::cout << "ConstantPadDriver: Backward verification passed." << std::endl;

    return miopenStatusSuccess;
}

#endif
