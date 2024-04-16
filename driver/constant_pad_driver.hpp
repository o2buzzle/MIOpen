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
#include <vector>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <../test/verify.hpp>

#define GET_NCDHW(n, c, d, h, w, idx, size) \
    {                                       \
        ulong ncdh = (idx) / size[4];       \
        w          = (idx) % size[4];       \
        ulong ncd  = ncdh / size[3];        \
        h          = ncdh % size[3];        \
        ulong nc   = ncd / size[2];         \
        d          = ncd % size[2];         \
        n          = nc / size[1];          \
        c          = nc % size[1];          \
    }

template <typename T>
T get5DValueAt(const T* x, const size_t* x_dims, size_t n, size_t c, size_t d, size_t h, size_t w)
{
    return x[n * x_dims[1] * x_dims[2] * x_dims[3] * x_dims[4] +
             c * x_dims[2] * x_dims[3] * x_dims[4] + d * x_dims[3] * x_dims[4] + h * x_dims[4] + w];
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

    auto input_strides = miopen::deref(inputDesc).GetStrides();

    size_t output_size =
        output_dims[0] * output_dims[1] * output_dims[2] * output_dims[3] * output_dims[4];

    for(size_t gid = 0; gid < output_size; ++gid)
    {
        bool flag = true;
        GET_NCDHW(o[0], o[1], o[2], o[3], o[4], gid, output_dims);

        for(int i = 0; i < 5; i++)
        {
            o[i] = o[i] - padding[2 * i];
            flag *= (o[i] < input_dims[i]);
        }

        if(flag)
        {
            output_host[gid] = get5DValueAt(input, input_dims.data(), o[0], o[1], o[2], o[3], o[4]);
        }
        else
        {
            output_host[gid] = value;
        }
    }
}

template <typename Tgpu, typename Tref>
class ConstantPadDriver : public Driver
{

public:
    ConstantPadDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&outputDesc);

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
    }

private:
    InputFlags inflags;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t outputDesc;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> out_dev;
    std::unique_ptr<GPUMem> padding_dev;

    std::vector<Tgpu> input;
    std::vector<Tgpu> output;
    std::vector<Tref> output_host;

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
    padding = GetPaddingsFromCmdLine();

    std::vector<int> input_dims = GetInputTensorLengthsFromCmdLine();
    SetTensorNd(inputDesc, input_dims, data_type);

    std::vector<int> output_dims;
    output_dims.reserve(input_dims.size());

    for(int i = 0; i < input_dims.size(); i++)
    {
        // TODO: ask the modnn team why it is like this.
        output_dims.push_back(input_dims[i] + 2 * padding[2 * i]);
    }

    SetTensorNd(outputDesc, output_dims, data_type);

    // Parse GetValueString -> (float)value
    auto value_str = inflags.GetValueStr("value");
    value          = strtof(value_str.c_str(), nullptr);

    return 0;
}

template <typename Tgpu, typename Tref>
int ConstantPadDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Only run the Forward Pass (Default=1)", "int");
    inflags.AddInputFlag("in_n", 'n', "3", "Input tensor dimension N", "int");
    inflags.AddInputFlag("in_c", 'c', "3", "Input tensor dimension C", "int");
    inflags.AddInputFlag("in_d", 'd', "0", "Input tensor dimension D", "int");
    inflags.AddInputFlag("in_h", 'k', "0", "Input tensor dimension H", "int");
    inflags.AddInputFlag("in_w", 'w', "0", "Input tensor dimension W", "int");
    inflags.AddInputFlag("pad_n", 'N', "0", "Padding dimension N", "int");
    inflags.AddInputFlag("pad_c", 'C', "0", "Padding dimension C", "int");
    inflags.AddInputFlag("pad_d", 'D', "0", "Padding dimension D", "int");
    inflags.AddInputFlag("pad_h", 'H', "0", "Padding dimension H", "int");
    inflags.AddInputFlag("pad_w", 'W', "0", "Padding dimension W", "int");
    inflags.AddInputFlag("value", 'v', "0", "Padding value", "float");

    inflags.AddInputFlag("iter", 'i', "1", "Number of iterations", "int");
    inflags.AddInputFlag("verify", 'V', "0", "Verify results", "int");
    inflags.AddInputFlag("time", 't', "0", "Enable/Disable time measurement", "int");
    inflags.AddInputFlag("wall", 'l', "0", "Enable/Disable walltime measurement", "int");

    return 0;
}

template <typename Tgpu, typename Tref>
std::vector<int> ConstantPadDriver<Tgpu, Tref>::GetInputTensorLengthsFromCmdLine()
{
    int in_n = inflags.GetValueInt("in_n");
    int in_c = inflags.GetValueInt("in_c");
    int in_d = inflags.GetValueInt("in_d");
    int in_h = inflags.GetValueInt("in_h");
    int in_w = inflags.GetValueInt("in_w");

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
    paddings[0]                  = inflags.GetValueInt("pad_n");
    paddings[2]                  = inflags.GetValueInt("pad_c");
    paddings[4]                  = inflags.GetValueInt("pad_d");
    paddings[6]                  = inflags.GetValueInt("pad_h");
    paddings[8]                  = inflags.GetValueInt("pad_w");
    return paddings;
}

template <typename Tgpu, typename Tref>
int ConstantPadDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t input_size  = GetTensorSize(inputDesc);
    size_t output_size = GetTensorSize(outputDesc);

    in_dev      = std::unique_ptr<GPUMem>(new GPUMem(0, input_size, sizeof(Tgpu)));
    out_dev     = std::unique_ptr<GPUMem>(new GPUMem(0, output_size, sizeof(Tgpu)));
    padding_dev = std::unique_ptr<GPUMem>(new GPUMem(0, 10, sizeof(size_t)));

    input       = std::vector<Tgpu>(input_size, static_cast<Tgpu>(0));
    output      = std::vector<Tgpu>(output_size, static_cast<Tgpu>(0));
    output_host = std::vector<Tref>(output_size, static_cast<Tref>(0));

    for(int i = 0; i < input.size(); i++)
    {
        input[i] = prng::gen_A_to_B(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }

    if(in_dev->ToGPU(GetStream(), input.data()) != 0)
        std::cerr << "ConstantPadDriver: Error copying data to GPU, size: " << in_dev->GetSize()
                  << std::endl;

    if(out_dev->ToGPU(GetStream(), output.data()) != 0)
        std::cerr << "ConstantPadDriver: Error copying data to GPU, size: " << out_dev->GetSize()
                  << std::endl;

    if(padding_dev->ToGPU(GetStream(), padding.data()) != 0)
        std::cerr << "ConstantPadDriver: Error copying data to GPU, size: "
                  << padding_dev->GetSize() << std::endl;

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
                             (const size_t*)padding_dev->GetMem(),
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
            std::cout << "Wall-clock time elapsed: " << t.gettime_ms() / iter << " ms" << std::endl;

        float kernel_avg_time = iter > 1 ? kernel_total_time / iter : kernel_first_time;
        std::cout << "Kernel time elapsed: " << kernel_avg_time << " ms" << std::endl;
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
int ConstantPadDriver<Tgpu, Tref>::RunBackwardGPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ConstantPadDriver<Tgpu, Tref>::VerifyBackward()
{
    return miopenStatusSuccess;
}

#endif
