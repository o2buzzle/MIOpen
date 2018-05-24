/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#include <miopen/gemm.hpp>
#include <miopen/handle.hpp>

namespace miopen {
// for debugging
#if 1
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vs)
{
    os << "{ size: " << vs.size() << ", entries: ";
    for(auto& v : vs)
        os << v << " ";
    os << "}";
    return os;
}
#endif

void CallGemm(Handle& handle,
              GemmDescriptor gemm_desc,
              const void* alpha,
              const void* A,
              int a_offset,
              const void* B,
              int b_offset,
              const void* beta,
              void* C,
              int c_offset,
              int find)
{
#if MIOPEN_USE_ROCBLAS
    std::cout << std::endl << __func__ << ": going to call rocblas" << std::endl;

    (void)find;

    if(!gemm_desc.isColMajor)
    {
        gemm_desc.isColMajor = true;
        std::swap(A, B);
        std::swap(a_offset, b_offset);
        std::swap(gemm_desc.transA, gemm_desc.transB);
        std::swap(gemm_desc.m, gemm_desc.n);
        std::swap(gemm_desc.lda, gemm_desc.ldb);
    }

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float alpha_local = *static_cast<const float*>(alpha);
    float beta_local  = *static_cast<const float*>(beta);
    hipEventRecord(start, nullptr);

    rocblas_sgemm(handle.rhandle.get(),
                  gemm_desc.transA ? rocblas_operation_transpose : rocblas_operation_none,
                  gemm_desc.transB ? rocblas_operation_transpose : rocblas_operation_none,
                  gemm_desc.m,
                  gemm_desc.n,
                  gemm_desc.k,
                  &alpha_local,
                  static_cast<const float*>(A) + a_offset,
                  gemm_desc.lda,
                  static_cast<const float*>(B) + b_offset,
                  gemm_desc.ldb,
                  &beta_local,
                  static_cast<float*>(C) + c_offset,
                  gemm_desc.ldc);

    hipEventRecord(stop, nullptr);
    hipDeviceSynchronize();
    float mS = 0;
    hipEventElapsedTime(&mS, start, stop);
    handle.ResetKernelTime();
    handle.AccumKernelTime(mS);

#elif MIOPEN_USE_MIOPENGEMM
    if(!gemm_desc.isColMajor)
    {
        std::swap(A, B);
        std::swap(a_offset, b_offset);
        std::swap(gemm_desc.transA, gemm_desc.transB);
        std::swap(gemm_desc.m, gemm_desc.n);
        std::swap(gemm_desc.lda, gemm_desc.ldb);
    }

    return miopen::try_([&] {
        miopen::GemmGeometry gg =
            miopen::CreateMIOpenGemmGeometry(gemm_desc.m,
                                             gemm_desc.n,
                                             gemm_desc.k,
                                             gemm_desc.lda,
                                             gemm_desc.ldb,
                                             gemm_desc.ldc,
                                             gemm_desc.transA,
                                             gemm_desc.transB,
                                             true,
                                             *(static_cast<const float*>(alpha)),
                                             *(static_cast<const float*>(beta)));

        if(find != 0)
        {
            gg.FindSolution(
                //.003, miopen::deref(handle), DataCast(A), DataCast(B), DataCast(C), false);
                60,
                miopen::deref(handle),
                DataCast(A),
                DataCast(B),
                DataCast(C),
                false);

            gg.RunGemm(miopen::deref(handle),
                       DataCast(A),
                       DataCast(B),
                       DataCast(C),
                       a_offset,
                       b_offset,
                       c_offset);
        }
        else
        {
            gg.RunGemm(miopen::deref(handle),
                       DataCast(A),
                       DataCast(B),
                       DataCast(C),
                       a_offset,
                       b_offset,
                       c_offset);
        }
    });
#else
    MIOPEN_THROW("No GEMM backend");
#endif
}

void CallGemmStridedBatched(Handle& handle,
                            GemmDescriptor gemm_desc,
                            const void* alpha,
                            const void* A,
                            int a_offset,
                            const void* B,
                            int b_offset,
                            const void* beta,
                            void* C,
                            int c_offset)
{
#if MIOPEN_USE_ROCBLAS
    std::cout << std::endl << __func__ << ": going to call rocblas" << std::endl;

    std::cout << __func__ << ": gemm_desc before swap" << gemm_desc << std::endl;

#if 1
    const float* A_old           = static_cast<const float*>(A);
    const float* B_old           = static_cast<const float*>(B);
    int a_offset_old             = a_offset;
    int b_offset_old             = b_offset;
    GemmDescriptor gemm_desc_old = gemm_desc;
#endif

    if(!gemm_desc.isColMajor)
    {
        gemm_desc.isColMajor = true;
        std::swap(A, B);
        std::swap(a_offset, b_offset);
        std::swap(gemm_desc.transA, gemm_desc.transB);
        std::swap(gemm_desc.m, gemm_desc.n);
        std::swap(gemm_desc.lda, gemm_desc.ldb);
        std::swap(gemm_desc.strideA, gemm_desc.strideB);
    }

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float alpha_local = *static_cast<const float*>(alpha);
    float beta_local  = *static_cast<const float*>(beta);

    std::cout << __func__ << ": alpha_local " << alpha_local << ", beta_local " << beta_local
              << std::endl;

    std::cout << __func__ << ": gemm_desc after swap" << gemm_desc << std::endl;

#if 1 // debug: output A, B, C
    {
        std::size_t a_sz = a_offset_old + gemm_desc_old.m * gemm_desc_old.k +
                           (gemm_desc_old.batch_count - 1) * gemm_desc_old.strideA;
        std::size_t b_sz = b_offset_old + gemm_desc_old.k * gemm_desc_old.n +
                           (gemm_desc_old.batch_count - 1) * gemm_desc_old.strideB;
        std::size_t c_sz = c_offset + gemm_desc_old.m * gemm_desc_old.n +
                           (gemm_desc_old.batch_count - 1) * gemm_desc_old.strideC;

        std::vector<float> tmp_a(a_sz, 0.);
        std::vector<float> tmp_b(b_sz, 0.);
        std::vector<float> tmp_c(c_sz, 0.);

        hipMemcpy(tmp_a.data(), A_old, a_sz * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(tmp_b.data(), B_old, b_sz * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(tmp_c.data(), C, c_sz * sizeof(float), hipMemcpyHostToDevice);

        std::cout << std::endl;
        // std::cout << __func__ << ": A before call rocblas: " << tmp_a << std::endl;
        // std::cout << __func__ << ": B before call rocblas: " << tmp_b << std::endl;
        // std::cout << __func__ << ": C before call rocblas: " << tmp_c << std::endl;

        float sum_c = std::accumulate(tmp_c.begin(), tmp_c.end(), float(0), std::plus<float>());
        std::cout << __func__ << ": sum_c before call rocblas" << sum_c << std::endl;
    }
#endif // debug: output A, B, C

    hipEventRecord(start, nullptr);
    rocblas_sgemm_strided_batched(
        handle.rhandle.get(),
        gemm_desc.transA ? rocblas_operation_transpose : rocblas_operation_none,
        gemm_desc.transB ? rocblas_operation_transpose : rocblas_operation_none,
        gemm_desc.m,
        gemm_desc.n,
        gemm_desc.k,
        &alpha_local,
        static_cast<const float*>(A) + a_offset,
        gemm_desc.lda,
        gemm_desc.strideA,
        static_cast<const float*>(B) + b_offset,
        gemm_desc.ldb,
        gemm_desc.strideB,
        &beta_local,
        static_cast<float*>(C) + c_offset,
        gemm_desc.ldc,
        gemm_desc.strideC,
        gemm_desc.batch_count);
    hipEventRecord(stop, nullptr);
    hipDeviceSynchronize();
    float mS = 0;
    hipEventElapsedTime(&mS, start, stop);
    handle.ResetKernelTime();
    handle.AccumKernelTime(mS);

#if 1 // debug: output A, B, C
    {
        std::size_t a_sz = a_offset_old + gemm_desc_old.m * gemm_desc_old.k +
                           (gemm_desc_old.batch_count - 1) * gemm_desc_old.strideA;
        std::size_t b_sz = b_offset_old + gemm_desc_old.k * gemm_desc_old.n +
                           (gemm_desc_old.batch_count - 1) * gemm_desc_old.strideB;
        std::size_t c_sz = c_offset + gemm_desc_old.m * gemm_desc_old.n +
                           (gemm_desc_old.batch_count - 1) * gemm_desc_old.strideC;

        std::vector<float> tmp_a(a_sz, 0.);
        std::vector<float> tmp_b(b_sz, 0.);
        std::vector<float> tmp_c(c_sz, 0.);

        hipMemcpy(tmp_a.data(), A_old, a_sz * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(tmp_b.data(), B_old, b_sz * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(tmp_c.data(), C, c_sz * sizeof(float), hipMemcpyHostToDevice);

        std::cout << std::endl;
        // std::cout << __func__ << ": A after call rocblas: " << tmp_a << std::endl;
        // std::cout << __func__ << ": B after call rocblas: " << tmp_b << std::endl;
        // std::cout << __func__ << ": C after call rocblas: " << tmp_c << std::endl;

        float sum_c = std::accumulate(tmp_c.begin(), tmp_c.end(), float(0), std::plus<float>());
        std::cout << __func__ << ": sum_c after call rocblas" << sum_c << std::endl;
    }
#endif // debug: output A, B, C
#else
    MIOPEN_THROW("No GEMM backend");
#endif
}

GemmDescriptor CreateGemmDescriptorConv1x1Fwd(const TensorDescriptor& xDesc,
                                              const TensorDescriptor& wDesc,
                                              const TensorDescriptor& yDesc)
{
    // [y] = [w] * [x]
    std::cout << std::endl << __func__ << std::endl;

    std::cout << __func__ << ": xDesc: " << xDesc << std::endl;
    std::cout << __func__ << ": wDesc: " << wDesc << std::endl;
    std::cout << __func__ << ": yDesc: " << yDesc << std::endl;

    GemmDescriptor gemm_desc;

    int in_n, in_c, in_h, in_w;
    std::tie(in_n, in_c, in_h, in_w) = tien<4>(xDesc.GetLengths());

    int wei_n;
    std::tie(wei_n, std::ignore, std::ignore, std::ignore) = tien<4>(wDesc.GetLengths());

    gemm_desc.isColMajor  = false;
    gemm_desc.transA      = false;
    gemm_desc.transB      = false;
    gemm_desc.m           = wei_n;
    gemm_desc.n           = in_h * in_w;
    gemm_desc.k           = in_c;
    gemm_desc.lda         = gemm_desc.k;
    gemm_desc.ldb         = gemm_desc.n;
    gemm_desc.ldc         = gemm_desc.n;
    gemm_desc.strideA     = 0;
    gemm_desc.strideB     = gemm_desc.k * gemm_desc.n;
    gemm_desc.strideC     = gemm_desc.m * gemm_desc.n;
    gemm_desc.batch_count = in_n;

    return gemm_desc;
}

GemmDescriptor CreateGemmDescriptorConv1x1BwdData(const TensorDescriptor& dyDesc,
                                                  const TensorDescriptor& wDesc,
                                                  const TensorDescriptor& dxDesc)
{
    // [dx] = transpose([w]) * [dy]
    std::cout << std::endl << __func__ << std::endl;

    std::cout << __func__ << ": dxDesc: " << dxDesc << std::endl;
    std::cout << __func__ << ": wDesc: " << wDesc << std::endl;
    std::cout << __func__ << ": dyDesc: " << dyDesc << std::endl;

    GemmDescriptor gemm_desc;

    int in_n, in_c, in_h, in_w;
    std::tie(in_n, in_c, in_h, in_w) = tien<4>(dxDesc.GetLengths());

    int wei_n;
    std::tie(wei_n, std::ignore, std::ignore, std::ignore) = tien<4>(wDesc.GetLengths());

    gemm_desc.isColMajor  = false;
    gemm_desc.transA      = true;
    gemm_desc.transB      = false;
    gemm_desc.m           = in_c;
    gemm_desc.n           = in_h * in_w;
    gemm_desc.k           = wei_n;
    gemm_desc.lda         = gemm_desc.m;
    gemm_desc.ldb         = gemm_desc.n;
    gemm_desc.ldc         = gemm_desc.n;
    gemm_desc.strideA     = 0;
    gemm_desc.strideB     = gemm_desc.k * gemm_desc.n;
    gemm_desc.strideC     = gemm_desc.m * gemm_desc.n;
    gemm_desc.batch_count = in_n;

    return gemm_desc;
}

std::ostream& operator<<(std::ostream& os, const GemmDescriptor& gemm_desc)
{
    os << "{ ";
    os << "isColMajor " << gemm_desc.isColMajor << ", ";
    os << "transA " << gemm_desc.transA << ", ";
    os << "transB " << gemm_desc.transB << ", ";
    os << "m " << gemm_desc.m << ", ";
    os << "n " << gemm_desc.n << ", ";
    os << "k " << gemm_desc.k << ", ";
    os << "lda " << gemm_desc.lda << ", ";
    os << "ldb " << gemm_desc.ldb << ", ";
    os << "ldc " << gemm_desc.ldc << ", ";
    os << "strideA " << gemm_desc.strideA << ", ";
    os << "strideB " << gemm_desc.strideB << ", ";
    os << "strideC " << gemm_desc.strideC << ", ";
    os << "batch_count " << gemm_desc.batch_count << " }";
    return os;
}

} // namespace miopen

#if MIOPEN_USE_MIOPENGEMM
namespace miopen {

GemmGeometry CreateGemmGeometryTranBwdData(const TensorDescriptor& dyDesc,
                                           const TensorDescriptor& wDesc,
                                           const TensorDescriptor& dxDesc,
                                           bool isDataColMajor,
                                           std::string& network_config)
{
    int in_n, in_c, in_h, in_w;
    std::tie(in_n, in_c, in_h, in_w) = tien<4>(dxDesc.GetLengths());

    int wei_c, wei_n, wei_h, wei_w;
    std::tie(wei_c, wei_n, wei_h, wei_w) = tien<4>(wDesc.GetLengths());

    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = tien<4>(dyDesc.GetLengths());

    // GEMM
    int K       = wei_n * wei_h * wei_w;
    int M       = wei_c;
    int N       = in_h * in_w;
    float alpha = 1.0;
    float beta  = 0.0;
    bool tA     = false;
    bool tB     = false;
    bool tC     = false;
    int lda     = K;
    int ldb     = N;
    int ldc     = N;

    MIOpenGEMM::Geometry tgg{};
    GemmGeometry gg;
    if(!isDataColMajor)
    {
        tgg = MIOpenGEMM::Geometry(true, tB, tA, tC, ldb, lda, ldc, N, M, K, 0, 'f');
        gg  = GemmGeometry{"miopenTransposeBwdDataAlgoGEMM", alpha, beta, tgg};
    }
    else
    {
        tgg = MIOpenGEMM::Geometry(false, tA, tB, tC, lda, ldb, ldc, M, N, K, 0, 'f');
        gg  = GemmGeometry{"miopenTransposeBwdDataAlgoGEMM", alpha, beta, tgg};
    }
    network_config = tgg.get_networkconfig_string();
    return gg;
}

GemmGeometry CreateGemmGeometryConvBwdData(const TensorDescriptor& dyDesc,
                                           const TensorDescriptor& wDesc,
                                           const TensorDescriptor& dxDesc,
                                           bool isDataColMajor,
                                           std::string& network_config)
{
    int in_n, in_c, in_h, in_w;
    std::tie(in_n, in_c, in_h, in_w) = tien<4>(dxDesc.GetLengths());

    int wei_n, wei_h, wei_w;
    std::tie(wei_n, std::ignore, wei_h, wei_w) = tien<4>(wDesc.GetLengths());

    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = tien<4>(dyDesc.GetLengths());

    // GEMM
    int K       = wei_n;
    int N       = out_h * out_w;
    int M       = in_c * wei_h * wei_w;
    float alpha = 1.0;
    float beta  = 0.0;
    bool tA     = true;
    bool tB     = false;
    bool tC     = false;
    int lda     = M;
    int ldb     = N;
    int ldc     = N;

    MIOpenGEMM::Geometry tgg{};
    GemmGeometry gg;
    if(!isDataColMajor)
    {
        tgg = MIOpenGEMM::Geometry(true, tB, tA, tC, ldb, lda, ldc, N, M, K, 0, 'f');
        gg  = GemmGeometry{"miopenConvolutionBwdDataAlgoGEMM", alpha, beta, tgg};
    }
    else
    {
        tgg = MIOpenGEMM::Geometry(false, tA, tB, tC, lda, ldb, ldc, M, N, K, 0, 'f');
        gg  = GemmGeometry{"miopenConvolutionBwdDataAlgoGEMM", alpha, beta, tgg};
    }
    network_config = tgg.get_networkconfig_string();
    return gg;
}

GemmGeometry CreateGemmGeometryConvBwdDataCNHW(const TensorDescriptor& dyDesc,
                                               const TensorDescriptor& wDesc,
                                               const TensorDescriptor& dxDesc,
                                               bool isDataColMajor,
                                               std::string& network_config)
{
    int in_n, in_c;
    std::tie(in_n, in_c, std::ignore, std::ignore) = tien<4>(dxDesc.GetLengths());

    int wei_n, wei_h, wei_w;
    std::tie(wei_n, std::ignore, wei_h, wei_w) = tien<4>(wDesc.GetLengths());

    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = tien<4>(dyDesc.GetLengths());

    // GEMM
    int K       = wei_n;
    int N       = in_n * out_h * out_w;
    int M       = in_c;
    float alpha = 1.0;
    float beta  = 0.0;
    bool tA     = true;
    bool tB     = false;
    bool tC     = false;
    int lda     = M;
    int ldb     = N;
    int ldc     = N;

    MIOpenGEMM::Geometry tgg{};
    GemmGeometry gg;
    if(!isDataColMajor)
    {
        tgg = MIOpenGEMM::Geometry(true, tB, tA, tC, ldb, lda, ldc, N, M, K, 0, 'f');
        gg  = GemmGeometry{"miopenConvolutionBwdDataAlgoGEMM", alpha, beta, tgg};
    }
    else
    {
        tgg = MIOpenGEMM::Geometry(false, tA, tB, tC, lda, ldb, ldc, M, N, K, 0, 'f');
        gg  = GemmGeometry{"miopenConvolutionBwdDataAlgoGEMM", alpha, beta, tgg};
    }
    network_config = tgg.get_networkconfig_string();
    return gg;
}

GemmGeometry CreateGemmGeometryConvBwdWeights(const TensorDescriptor& dyDesc,
                                              const TensorDescriptor& xDesc,
                                              const TensorDescriptor& dwDesc,
                                              bool isDataColMajor,
                                              std::string& network_config)
{
    int in_n, in_c, in_h, in_w;
    std::tie(in_n, in_c, in_h, in_w) = tien<4>(xDesc.GetLengths());

    int wei_n, wei_h, wei_w;
    std::tie(wei_n, std::ignore, wei_h, wei_w) = tien<4>(dwDesc.GetLengths());

    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = tien<4>(dyDesc.GetLengths());

    // GEMM
    int N       = in_c * wei_h * wei_w;
    int M       = wei_n;
    int K       = out_h * out_w;
    bool tA     = false;
    bool tB     = true;
    bool tC     = false;
    int lda     = K;
    int ldb     = K;
    int ldc     = N;
    float alpha = 1.0;
    float beta  = 1.0;

    MIOpenGEMM::Geometry tgg{};
    GemmGeometry gg;
    if(!isDataColMajor)
    {
        tgg = MIOpenGEMM::Geometry(true, tB, tA, tC, ldb, lda, ldc, N, M, K, 0, 'f');
        gg  = GemmGeometry{"miopenConvolutionBwdWeightsAlgoGEMM", alpha, beta, tgg};
    }
    else
    {
        tgg = MIOpenGEMM::Geometry(true, tA, tB, tC, lda, ldb, ldc, M, N, K, 0, 'f');
        gg  = GemmGeometry{"miopenConvolutionBwdWeightsAlgoGEMM", alpha, beta, tgg};
    }
    network_config = tgg.get_networkconfig_string();
    return gg;
}

GemmGeometry CreateGemmGeometryConvFwd(const TensorDescriptor& xDesc,
                                       const TensorDescriptor& wDesc,
                                       const TensorDescriptor& yDesc,
                                       bool isDataColMajor,
                                       std::string& network_config)
{
    int in_n, in_c, in_h, in_w;
    std::tie(in_n, in_c, in_h, in_w) = tien<4>(xDesc.GetLengths());

    int wei_n, wei_h, wei_w;
    std::tie(wei_n, std::ignore, wei_h, wei_w) = tien<4>(wDesc.GetLengths());

    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = tien<4>(yDesc.GetLengths());

    // GEMM
    int K       = in_c * wei_h * wei_w;
    int M       = wei_n;
    int N       = out_h * out_w;
    float alpha = 1.0;
    float beta  = 0.0;
    bool tA     = false;
    bool tB     = false;
    bool tC     = false;
    int lda     = K;
    int ldb     = N;
    int ldc     = N;

    MIOpenGEMM::Geometry tgg{};
    GemmGeometry gg;
    if(!isDataColMajor)
    {
        tgg = MIOpenGEMM::Geometry(true, tB, tA, tC, ldb, lda, ldc, N, M, K, 0, 'f');
        gg  = GemmGeometry{"miopenConvolutionFwdAlgoGEMM", alpha, beta, tgg};
    }
    else
    {
        tgg = MIOpenGEMM::Geometry(true, tA, tB, tC, lda, ldb, ldc, M, N, K, 0, 'f');
        gg  = GemmGeometry{"miopenConvolutionFwdAlgoGEMM", alpha, beta, tgg};
    }
    network_config = tgg.get_networkconfig_string();
    return gg;
}

GemmGeometry CreateGemmGeometryConvFwdCNHW(const TensorDescriptor& xDesc,
                                           const TensorDescriptor& wDesc,
                                           const TensorDescriptor& yDesc,
                                           bool isDataColMajor,
                                           std::string& network_config)
{
    int in_n, in_c;
    std::tie(in_n, in_c, std::ignore, std::ignore) = tien<4>(xDesc.GetLengths());

    int wei_n;
    std::tie(wei_n, std::ignore, std::ignore, std::ignore) = tien<4>(wDesc.GetLengths());

    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = tien<4>(yDesc.GetLengths());

    // GEMM
    int K       = in_c;
    int M       = wei_n;
    int N       = in_n * out_h * out_w;
    float alpha = 1.0;
    float beta  = 0.0;
    bool tA     = false;
    bool tB     = false;
    bool tC     = false;
    int lda     = K;
    int ldb     = N;
    int ldc     = N;

    MIOpenGEMM::Geometry tgg{};
    GemmGeometry gg;
    if(!isDataColMajor)
    {
        tgg = MIOpenGEMM::Geometry(true, tB, tA, tC, ldb, lda, ldc, N, M, K, 0, 'f');
        gg  = GemmGeometry{"miopenConvolutionFwdAlgoGEMM", alpha, beta, tgg};
    }
    else
    {
        tgg = MIOpenGEMM::Geometry(true, tA, tB, tC, lda, ldb, ldc, M, N, K, 0, 'f');
        gg  = GemmGeometry{"miopenConvolutionFwdAlgoGEMM", alpha, beta, tgg};
    }
    network_config = tgg.get_networkconfig_string();
    return gg;
}

GemmGeometry CreateMIOpenGemmGeometry(int M,
                                      int N,
                                      int K,
                                      int lda,
                                      int ldb,
                                      int ldc,
                                      bool tA,
                                      bool tB,
                                      bool isDataColMajor,
                                      float alpha,
                                      float beta)
{
    MIOpenGEMM::Geometry tgg{};

    // Assuming we are using miopengemm as only col major
    // Therefore, if the user provides data in col. major
    // then no transformations are requrired and vice versa
    if(isDataColMajor)
    {
        tgg = MIOpenGEMM::Geometry(
            true, tA, tB, false, lda, ldb, ldc, M, N, K, 0, 'f'); // jn : added 0 for no workspace,
                                                                  // 'f' for single prec.

        return GemmGeometry{"miopenGEMM", alpha, beta, tgg};
    }
    else
    {
        tgg = MIOpenGEMM::Geometry(
            true, tB, tA, false, ldb, lda, ldc, N, M, K, 0, 'f'); // jn : added 0 for no workspace,
                                                                  // 'f' for single prec.

        return GemmGeometry{"miopenGEMM", alpha, beta, tgg};
    }
}

GemmGeometry GetGemmGeometry(Handle& handle, std::string algorithm_name, std::string network_config)
{
    auto gemm_iterator = handle.geo_map.find(std::make_pair(algorithm_name, network_config));
    if(gemm_iterator != handle.geo_map.end())
    {
        return *gemm_iterator->second;
    }
    else
    {
        MIOPEN_THROW("looking for gemm kernel (does not exist): " + algorithm_name + ", " +
                     network_config);
    }
}

GemmGeometry CreateGemmGeometryRNN(int M,
                                   int N,
                                   int K,
                                   float alpha,
                                   float beta,
                                   bool tA,
                                   bool tB,
                                   bool tC,
                                   int lda,
                                   int ldb,
                                   int ldc,
                                   bool isDataColMajor,
                                   std::string& network_config)
{
    // GEMM
    MIOpenGEMM::Geometry tgg{};
    GemmGeometry gg;
    (void)isDataColMajor;

    tgg = MIOpenGEMM::Geometry(false, tA, tB, tC, lda, ldb, ldc, M, N, K, 0, 'f');
    gg  = GemmGeometry{"miopenRNNAlgoGEMM", alpha, beta, tgg};

    network_config = tgg.get_networkconfig_string();
    return gg;
}

GemmGeometry ScanGemmGeometryRNN(Handle& handle,
                                 ConstData_t A,
                                 ConstData_t B,
                                 Data_t C,
                                 int M,
                                 int N,
                                 int K,
                                 float alpha,
                                 float beta,
                                 bool tA,
                                 bool tB,
                                 bool tC,
                                 int lda,
                                 int ldb,
                                 int ldc,
                                 bool isDataColMajor,
                                 std::string& network_config,
                                 float timeout)
{

    auto gg = CreateGemmGeometryRNN(
        M, N, K, alpha, beta, tA, tB, tC, lda, ldb, ldc, isDataColMajor, network_config);

    auto gemm_iterator = handle.geo_map.find(std::make_pair("miopenRNNAlgoGEMM", network_config));
    if(gemm_iterator != handle.geo_map.end())
    {
        gg = *gemm_iterator->second;
    }
    else
    {
        gg.FindSolution(timeout, handle, A, B, C, false);
    }

    return gg;
}

void RunGemmGeometryRNN(Handle& handle,
                        ConstData_t A,
                        ConstData_t B,
                        Data_t C,
                        int M,
                        int N,
                        int K,
                        float alpha,
                        float beta,
                        bool tA,
                        bool tB,
                        bool tC,
                        int lda,
                        int ldb,
                        int ldc,
                        int a_offset,
                        int b_offset,
                        int c_offset,
                        bool isDataColMajor,
                        std::string& network_config,
                        float timeout)
{

    auto gg = ScanGemmGeometryRNN(handle,
                                  A,
                                  B,
                                  C,
                                  M,
                                  N,
                                  K,
                                  alpha,
                                  beta,
                                  tA,
                                  tB,
                                  tC,
                                  lda,
                                  ldb,
                                  ldc,
                                  isDataColMajor,
                                  network_config,
                                  timeout);

    gg.RunGemm(handle, A, B, C, a_offset, b_offset, c_offset);
}

} // namespace miopen
#endif // MIOPEN_USE_MIOPENGEMM
