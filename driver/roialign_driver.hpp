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
#ifndef GUARD_MIOPEN_ROIALIGN_DRIVER_HPP
#define GUARD_MIOPEN_ROIALIGN_DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include "random.hpp"
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdlib>
#include <iomanip>
#include <memory>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <numeric>
#include <vector>
#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>

inline std::vector<std::string> split(const std::string& s, char delim)
{
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;

    while(getline(ss, item, delim))
    {
        result.push_back(item);
    }

    return result;
}
template <typename Tgpu, typename Tref>
class RoIAlignDriver : public Driver
{

public:
    RoIAlignDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&roisDesc);
        miopenCreateTensorDescriptor(&outputDesc);

        data_type = miopen_type<Tgpu>{};
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;
    int ParseRoIs(std::vector<Tgpu>& rois, const std::string& rois_str);

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    int VerifyForward() override;
    int VerifyBackward() override;

    ~RoIAlignDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(roisDesc);
        miopenDestroyTensorDescriptor(outputDesc);
    }

private:
    InputFlags inflags;

    int forw;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t roisDesc;
    miopenTensorDescriptor_t outputDesc;
    size_t output_h;
    size_t output_w;

    float spatial_scale;
    int sampling_ratio;
    bool aligned;
    int roi_batch_idx;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> rois_dev;
    std::unique_ptr<GPUMem> out_dev;

    std::vector<Tgpu> in_host;
    std::vector<Tgpu> rois_host;

    std::vector<Tgpu> out_host;
    std::vector<Tref> out_ref;
};

template <typename Tgpu, typename Tref>
int RoIAlignDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Only run forward pass (Default=1)", "int");
    inflags.AddTensorFlag("input", 'I', "1x3x224x224");
    inflags.AddInputFlag("rois",
                         'r',
                         "0-0-0-100-100,0-12-12-200-200,0-24-24-150-150",
                         "RoIs (format: elem_idx-x1-y1-x2-y2,elem_idx-x1-y1-x2-y2)",
                         "string");
    inflags.AddInputFlag("output_h", 'H', "24", "Output Height (Default=244)", "int");
    inflags.AddInputFlag("output_w", 'W', "24", "Output Width (Default=244)", "int");
    inflags.AddInputFlag("spatial_scale", 's', "0.5", "Spatial Scale (Default=0.0625)", "float");
    inflags.AddInputFlag("sampling_ratio", 'S', "2", "Sampling Ratio (Default=1)", "int");
    inflags.AddInputFlag("aligned", 'a', "0", "Aligned (Default=0)", "int");
    inflags.AddInputFlag("roi_batch_idx", 'B', "0", "RoI Batch Index (Default=0)", "int");

    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RoIAlignDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RoIAlignDriver<Tgpu, Tref>::ParseRoIs(std::vector<Tgpu>& rois, const std::string& rois_str)
{
    std::vector<std::string> rois_vec = split(rois_str, ',');
    std::vector<int> new_len          = {rois_vec.size(), 5};
    SetTensorNd(roisDesc, new_len, data_type);
    rois.resize(rois_vec.size() * 5);

    for(int i = 0; i < rois_vec.size(); i++)
    {
        std::vector<std::string> elem_vec = split(rois_vec[i], '-');
        assert(elem_vec.size() == 5);
        for(int j = 0; j < 5; j++)
        {
            rois[i * 5 + j] = std::stof(elem_vec[j]);
        }
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RoIAlignDriver<Tgpu, Tref>::GetandSetData()
{
    std::vector<int> in_len = inflags.GetValueTensor("input").lengths;
    SetTensorNd(inputDesc, in_len, data_type);

    auto status = ParseRoIs(rois_host, inflags.GetValueStr("rois"));
    assert(status == miopenStatusSuccess);

    output_h = inflags.GetValueInt("output_h");
    output_w = inflags.GetValueInt("output_w");

    auto k                   = GetTensorLengths(roisDesc)[0];
    auto c                   = in_len[1];
    auto h                   = output_w;
    auto w                   = output_h;
    std::vector<int> out_len = {k, c, h, w};

    SetTensorNd(outputDesc, out_len, data_type);

    spatial_scale  = inflags.GetValueDouble("spatial_scale");
    sampling_ratio = inflags.GetValueInt("sampling_ratio");
    aligned        = inflags.GetValueInt("aligned") == 1;
    roi_batch_idx  = inflags.GetValueInt("roi_batch_idx");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RoIAlignDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t in_sz  = GetTensorSize(inputDesc);
    size_t out_sz = GetTensorSize(outputDesc);

    uint32_t ctx = 0;

    in_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, GetTensorSize(inputDesc), sizeof(Tgpu)));
    rois_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, GetTensorSize(roisDesc), sizeof(Tgpu)));
    out_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, GetTensorSize(outputDesc), sizeof(Tgpu)));

    in_host  = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(1));
    out_host = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    out_ref  = std::vector<Tref>(out_sz, static_cast<Tref>(0));

    /*
    Fill in a predictable way
    for i in range(224):
        for j in range(224):
            input[0, 0, i, j] = i / 224
            input[0, 1, i, j] = j / 224
            input[0, 2, i, j] = (i - 112) / 112
    */

    for(int i = 0; i < 224; i++)
    {
        for(int j = 0; j < 224; j++)
        {
            in_host[0 * 224 * 224 + i * 224 + j] = i / 224.0f;
            in_host[1 * 224 * 224 + i * 224 + j] = j / 224.0f;
            in_host[2 * 224 * 224 + i * 224 + j] = (i - 112) / 112.0f;
        }
    }

    if(in_dev->ToGPU(GetStream(), in_host.data()) != 0)
        std::cerr << "Error copying (in) to GPU, size: " << in_dev->GetSize() << std::endl;

    if(out_dev->ToGPU(GetStream(), out_host.data()) != 0)
        std::cerr << "Error copying (out) to GPU, size: " << out_dev->GetSize() << std::endl;

    if(rois_dev->ToGPU(GetStream(), rois_host.data()) != 0)
        std::cerr << "Error copying (rois) to GPU, size: " << rois_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RoIAlignDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenRoIAlignForward(GetHandle(),
                              inputDesc,
                              in_dev->GetMem(),
                              roisDesc,
                              rois_dev->GetMem(),
                              outputDesc,
                              out_dev->GetMem(),
                              output_h,
                              output_w,
                              spatial_scale,
                              sampling_ratio,
                              aligned,
                              roi_batch_idx);

        float time = 0.0f;
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
            std::cout << "Wall-clock Time Forward Sum Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;

        std::cout << "GPU Kernel Time Forward Sum Elapsed: " << kernel_average_time << " ms\n";
    }

    if(out_dev->FromGPU(GetStream(), out_host.data()) != 0)
        std::cerr << "Error copying (out_dev) from GPU, size: " << out_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RoIAlignDriver<Tgpu, Tref>::RunForwardCPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RoIAlignDriver<Tgpu, Tref>::RunBackwardGPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RoIAlignDriver<Tgpu, Tref>::RunBackwardCPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RoIAlignDriver<Tgpu, Tref>::VerifyForward()
{
    // for now, dump to stdout to check with torch
    for(int i = 0; i < out_host.size(); i++)
    {
        // std::cout << std::setprecision(4) << "in[" << i << "] = " << in_host[i] << std::endl;
        std::cout << std::setprecision(4) << "out[" << i << "] = " << out_host[i] << std::endl;
    }
    std::cout << "\n";

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RoIAlignDriver<Tgpu, Tref>::VerifyBackward()
{
    return miopenStatusSuccess;
}

#endif
