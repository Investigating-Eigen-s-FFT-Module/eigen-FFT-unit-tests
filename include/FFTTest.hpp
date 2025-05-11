#pragma once
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <random>

#include "options.hpp"
#include "test_types.hpp"

template <typename CallSpec>
class FFTTest : public testing::Test {
 protected:
  FFTTest() { std::srand(42); }

  using SrcMatrixType = typename CallSpec::SrcMatrixType;
  using DstMatrixType = typename CallSpec::DstMatrixType;
  static constexpr int Options = CallSpec::Options;

  // TODO: different test to see if prealloc of dst is respected
  void GenerateTestdata() {
    if constexpr (CallSpec::Is1D) {
      if constexpr (CallSpec::SrcIsDynamic) {
        src = SrcMatrixType::Random(10);
      } else {
        src = SrcMatrixType::Random();
      }
    } else {
      if constexpr (CallSpec::SrcIsDynamic) {
        src = SrcMatrixType::Random(10, 10);
      } else {
        src = SrcMatrixType::Random();
      }
    }
  }

  void TestBinaryTransform() {
    // C2C
    if constexpr (!CallSpec::RealTransform) {
      Eigen::FFT<Options> fft;
      if constexpr (CallSpec::Forward) {
        fft.fwd(dst, src);
        TestAgainstOracle(dst, src, src.rows(), src.cols());
      }
    }
  }

 private:
  SrcMatrixType src;
  DstMatrixType dst;

  DstMatrixType OracleFFT(const SrcMatrixType& src, const size_t nfft0, const size_t nfft1) {
    using Scalar = typename SrcMatrixType::Scalar;
    using RealScalar = typename SrcMatrixType::RealScalar;
    using ComplexScalar = std::complex<RealScalar>;
    using BigRealScalar = long double;
    using BigComplexScalar = std::complex<BigRealScalar>;

    Eigen::MatrixX<BigComplexScalar> src_cp = src.template cast<BigComplexScalar>();
    BigComplexScalar* src_ptr = src_cp.data();

    using namespace pocketfft;

    const shape_t shape{nfft0, nfft1};
    const shape_t axes{0, 1};
    const stride_t stride{static_cast<ptrdiff_t>(sizeof(BigComplexScalar) * nfft1),
                          static_cast<ptrdiff_t>(sizeof(BigComplexScalar))};

    Eigen::MatrixX<BigComplexScalar> dst(nfft0, nfft1);
    BigComplexScalar* dst_ptr = dst.data();

    c2c(shape, stride, stride, axes, CallSpec::Forward, src_ptr, dst_ptr, static_cast<BigRealScalar>(1),
        static_cast<size_t>(1));

    DstMatrixType dst_cp = dst.cast<typename DstMatrixType::Scalar>();

    return dst_cp;
  }

  // NOTE: expects src to be the full size even when HalfSpectrum is enabled and it's C2R
  void TestAgainstOracle(DstMatrixType& dst, SrcMatrixType& src, size_t nfft0, size_t nfft1) {
    using namespace Eigen::FFTOption;

    auto oracle = OracleFFT(src, nfft0, nfft1);

    // IFFT(FFT(x)) == a * x if Unscaled is enabled
    if constexpr (Options & Unscaled) {
      dst.normalize();
      oracle.normalize();
    } else {
      // No scaling of FFT(x) atm
      if constexpr (CallSpec::Forward) {
        dst.normalize();
        oracle.normalize();
      } else {
        // Mimic scaling done by Eigen::FFT
        oracle /= oracle.size();
      }
    }

    // Conditions under which reduced array sizes can happen
    if constexpr ((Options & HalfSpectrum) && CallSpec::RealTransform) {
      if constexpr (CallSpec::Forward) {
        if constexpr (CallSpec::Is1D) {
          ASSERT_EQ(dst.size(), src.size() / 2 + 1);

          oracle.resize(src.size() / 2 + 1);

          ASSERT_TRUE(dst.isApprox(oracle));
        } else {
          ASSERT_EQ(dst.rows(), src.rows() / 2 + 1);

          oracle.resize(src.rows() / 2 + 1, src.cols());

          ASSERT_TRUE(dst.isApprox(oracle));
        }
      } else {
        ASSERT_TRUE(dst.isApprox(oracle.real()));
      }
    } else {
      ASSERT_TRUE(dst.isApprox(oracle));
    }
  }
};