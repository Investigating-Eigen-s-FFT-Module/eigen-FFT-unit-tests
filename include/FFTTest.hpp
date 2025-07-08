#pragma once
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <random>
#include <unsupported/Eigen/FFT>

#include "constants.hpp"
#include "options.hpp"
#include "test_types.hpp"

template <typename CallSpec>
class FFTTestBase : public testing::Test {
 protected:
  FFTTestBase() { std::srand(42); }

  using SrcMatrixType = typename CallSpec::SrcMatrixType;
  using DstMatrixType = typename CallSpec::DstMatrixType;
  static constexpr int Options = CallSpec::Options;

  virtual void GenerateTestdata() {
    if constexpr (CallSpec::Is1D) {
      if constexpr (CallSpec::SrcIsDynamic) {
        src = SrcMatrixType::Random(NFFT0);
      } else {
        src = SrcMatrixType::Random();
      }
    } else {
      if constexpr (CallSpec::SrcIsDynamic) {
        src = SrcMatrixType::Random(NFFT0, NFFT1);
      } else {
        src = SrcMatrixType::Random();
      }
    }
  }

  virtual void TestBinaryTransform() = 0;
  virtual void TestBinaryTransform(Eigen::Index nfft0, Eigen::Index nfft1) = 0;
  virtual void TestBinaryTransformExplicitCompileTimeNFFT() = 0;

  virtual void TestUnaryTransform() = 0;
  virtual void TestUnaryTransform(Eigen::Index nfft0, Eigen::Index nfft1) = 0;
  virtual void TestUnaryTransformExplicitCompileTimeNFFT() = 0;
  virtual void TestUnaryTransformExplicitDstType() = 0;

  virtual void TestAgainstOracle(DstMatrixType& dst, SrcMatrixType& src, size_t nfft0, size_t nfft1) = 0;

  SrcMatrixType src;
  DstMatrixType dst;

  using Scalar = typename SrcMatrixType::Scalar;
  using RealScalar = typename SrcMatrixType::RealScalar;
  using ComplexScalar = std::complex<RealScalar>;
  using BigRealScalar = long double;
  using BigComplexScalar = std::complex<BigRealScalar>;

  // Templated src type in case src is static:
  // Casting to BigComplexScalar with static src type may require resizing, which is illegal.
  // The template allows specifying a dynamic equivalent wherever necessary
  template <typename AdjustedSrcMatrixType>
  auto OracleFFT(const AdjustedSrcMatrixType& src, const size_t nfft0, const size_t nfft1) {
    Eigen::MatrixX<BigComplexScalar> src_cp = src.template cast<BigComplexScalar>();
    BigComplexScalar* src_ptr = src_cp.data();
    Eigen::MatrixX<BigComplexScalar> dst_cp(nfft0, nfft1);

    using namespace pocketfft;

    const shape_t shape{nfft0, nfft1};
    const shape_t axes{1, 0};
    const stride_t stride_in{static_cast<ptrdiff_t>(src_cp.rowStride() * sizeof(BigComplexScalar)),
                             static_cast<ptrdiff_t>(src_cp.colStride() * sizeof(BigComplexScalar))};
    const stride_t stride_out{static_cast<ptrdiff_t>(dst_cp.rowStride() * sizeof(BigComplexScalar)),
                              static_cast<ptrdiff_t>(dst_cp.colStride() * sizeof(BigComplexScalar))};

    BigComplexScalar* dst_ptr = dst_cp.data();

    c2c(shape, stride_in, stride_out, axes, CallSpec::Forward, src_ptr, dst_ptr, static_cast<BigRealScalar>(1),
        static_cast<size_t>(1));

    return dst_cp;
  }

  template <typename AdjustedSrcMatrixType>
  auto OracleFFT(const AdjustedSrcMatrixType& src, const size_t nfft) {
    using ComplexVectorType = std::conditional_t<AdjustedSrcMatrixType::ColsAtCompileTime == 1,
                                                 Eigen::RowVectorX<BigComplexScalar>, Eigen::VectorX<BigComplexScalar>>;
    ComplexVectorType src_cp = src.template cast<BigComplexScalar>();
    BigComplexScalar* src_ptr = src_cp.data();
    ComplexVectorType dst_cp(nfft);

    using namespace pocketfft;

    const shape_t shape{nfft};
    const shape_t axes{0};
    const stride_t stride_in{static_cast<ptrdiff_t>(src_cp.stride() * sizeof(BigComplexScalar))};
    const stride_t stride_out{static_cast<ptrdiff_t>(dst_cp.stride() * sizeof(BigComplexScalar))};

    BigComplexScalar* dst_ptr = dst_cp.data();

    c2c(shape, stride_in, stride_out, axes, CallSpec::Forward, src_ptr, dst_ptr, static_cast<BigRealScalar>(1),
        static_cast<size_t>(1));
    return dst_cp;
  }

  template <typename AdjustedSrcMatrixType>
  auto GetOracle(const AdjustedSrcMatrixType& src, const size_t nfft0, const size_t nfft1) {
    if constexpr (CallSpec::Is1D) {
      return OracleFFT(src, std::max(nfft0, nfft1));
    } else {
      return OracleFFT(src, nfft0, nfft1);
    }
  }
};

template <typename CallSpec>
class C2CTest : public FFTTestBase<CallSpec> {
  using Base = FFTTestBase<CallSpec>;
  using Base::dst;
  using Base::Options;
  using Base::src;
  using typename Base::DstMatrixType;
  using typename Base::SrcMatrixType;

 protected:
  C2CTest() : FFTTestBase<CallSpec>() { static_assert(!CallSpec::RealTransform); }

  virtual void TestBinaryTransform() override final {
    using namespace Eigen::FFTOption;
    Eigen::FFT<Options> fft;
    if constexpr (CallSpec::Forward) {
      fft.fwd(dst, src);
      TestAgainstOracle(dst, src, src.rows(), src.cols());
    } else {
      if constexpr (CallSpec::Options & Eigen::FFTOption::HalfSpectrum) {
        // HalfSpectrum inv always requires specifying nfft
        if constexpr (CallSpec::Is1D) {
          fft.inv(dst, src, NFFT0);
        } else {
          fft.inv(dst, src, NFFT0, NFFT1);
        }
      }
      TestAgainstOracle(dst, src, src.rows(), src.cols());
    }
  }

  virtual void TestBinaryTransform(Eigen::Index nfft0, Eigen::Index nfft1) override final {
    using namespace Eigen::FFTOption;
    Eigen::FFT<Options> fft;
    if constexpr (CallSpec::Forward) {
      if constexpr (CallSpec::Is1D) {
        fft.fwd(dst, src, nfft0);
      } else {
        fft.fwd(dst, src, nfft0, nfft1);
      }
      TestAgainstOracle(dst, src, src.rows(), src.cols());
    } else {
      if constexpr (CallSpec::Is1D) {
        fft.inv(dst, src, nfft0);
      } else {
        fft.inv(dst, src, nfft0, nfft1);
      }
      TestAgainstOracle(dst, src, src.rows(), src.cols());
    }
  }

  virtual void TestBinaryTransformExplicitCompileTimeNFFT() override final {
    using namespace Eigen::FFTOption;
    Eigen::FFT<Options> fft;
    if constexpr (CallSpec::Forward) {
      if constexpr (CallSpec::Is1D) {
        // fft.template fwd<DstMatrixType, SrcMatrixType, NFFT0>(dst, src);
        fft.template fwd<NFFT0>(dst, src);
      } else {
        // fft.template fwd<DstMatrixType, SrcMatrixType, NFFT0, NFFT1>(dst, src);
        fft.template fwd<NFFT0, NFFT1>(dst, src);
      }
      TestAgainstOracle(dst, src, src.rows(), src.cols());
    } else {
      if constexpr (CallSpec::Is1D) {
        // fft.template inv<DstMatrixType, SrcMatrixType, NFFT0>(dst, src);
        fft.template inv<NFFT0>(dst, src);
      } else {
        // fft.template inv<DstMatrixType, SrcMatrixType, NFFT0, NFFT1>(dst, src);
        fft.template inv<NFFT0, NFFT1>(dst, src);
      }
      TestAgainstOracle(dst, src, src.rows(), src.cols());
    }
  }

  virtual void TestUnaryTransform() override final {
    using namespace Eigen::FFTOption;
    Eigen::FFT<Options> fft;
    if constexpr (CallSpec::Forward) {
      dst = fft.fwd(src);
      TestAgainstOracle(dst, src, src.rows(), src.cols());
    } else {
      // HalfSpectrum inv always requires specifying nfft
      if constexpr (CallSpec::Options & Eigen::FFTOption::HalfSpectrum) {
        if constexpr (CallSpec::Is1D) {
          dst = fft.inv(src, NFFT0);
        } else {
          dst = fft.inv(src, NFFT0, NFFT1);
        }
      } else {
        dst = fft.inv(src);
      }
      TestAgainstOracle(dst, src, src.rows(), src.cols());
    }
  }

  virtual void TestUnaryTransform(Eigen::Index nfft0, Eigen::Index nfft1) override final {
    using namespace Eigen::FFTOption;
    Eigen::FFT<Options> fft;
    if constexpr (CallSpec::Forward) {
      if constexpr (CallSpec::Is1D) {
        dst = fft.fwd(src, nfft0);
      } else {
        dst = fft.fwd(src, nfft0, nfft1);
      }
      TestAgainstOracle(dst, src, src.rows(), src.cols());
    } else {
      if constexpr (CallSpec::Is1D) {
        dst = fft.inv(src, nfft0);
      } else {
        dst = fft.inv(src, nfft0, nfft1);
      }
      TestAgainstOracle(dst, src, src.rows(), src.cols());
    }
  }

  virtual void TestUnaryTransformExplicitCompileTimeNFFT() override final {
    using namespace Eigen::FFTOption;
    Eigen::FFT<Options> fft;
    if constexpr (CallSpec::Forward) {
      if constexpr (CallSpec::Is1D) {
        dst = fft.template fwd<NFFT0>(src);
      } else {
        dst = fft.template fwd<NFFT0, NFFT1>(src);
      }
      TestAgainstOracle(dst, src, src.rows(), src.cols());
    } else {
      if constexpr (CallSpec::Is1D) {
        dst = fft.template inv<NFFT0>(src);
      } else {
        dst = fft.template inv<NFFT0, NFFT1>(src);
      }
      TestAgainstOracle(dst, src, src.rows(), src.cols());
    }
  }

  virtual void TestUnaryTransformExplicitDstType() override final {
    using namespace Eigen::FFTOption;
    Eigen::FFT<Options> fft;
    if constexpr (CallSpec::Forward) {
      dst = fft.template fwd<DstMatrixType>(src);
      TestAgainstOracle(dst, src, src.rows(), src.cols());
    } else {
      if constexpr (CallSpec::DstIsDynamic) {
        if constexpr (CallSpec::Is1D) {
          dst = fft.template inv<DstMatrixType, NFFT0>(src);
        } else {
          dst = fft.template inv<DstMatrixType, NFFT0, NFFT1>(src);
        }
      } else {
        dst = fft.template inv<DstMatrixType>(src);
      }
      TestAgainstOracle(dst, src, src.rows(), src.cols());
    }
  }

  virtual void TestAgainstOracle(DstMatrixType& dst, SrcMatrixType& src, size_t nfft0, size_t nfft1) override final {
    {
      using namespace Eigen::FFTOption;

      DstMatrixType oracle = Base::GetOracle(src, nfft0, nfft1).template cast<typename DstMatrixType::Scalar>();

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
    }
  }
};

template <typename CallSpec>
class R2CTest : public FFTTestBase<CallSpec> {
  using Base = FFTTestBase<CallSpec>;
  using Base::dst;
  using Base::Options;
  using Base::src;
  using typename Base::DstMatrixType;
  using typename Base::SrcMatrixType;

 protected:
  R2CTest() : FFTTestBase<CallSpec>() { static_assert(CallSpec::RealTransform && CallSpec::Forward); }

  virtual void TestBinaryTransform() override final {
    using namespace Eigen::FFTOption;
    Eigen::FFT<Options> fft;
    fft.fwd(dst, src);
    TestAgainstOracle(dst, src, src.rows(), src.cols());
  }

  virtual void TestBinaryTransform(Eigen::Index nfft0, Eigen::Index nfft1) override final {
    using namespace Eigen::FFTOption;
    Eigen::FFT<Options> fft;
    if constexpr (CallSpec::Is1D) {
      fft.fwd(dst, src, nfft0);
    } else {
      fft.fwd(dst, src, nfft0, nfft1);
    }
    TestAgainstOracle(dst, src, src.rows(), src.cols());
  }

  virtual void TestBinaryTransformExplicitCompileTimeNFFT() override final {
    using namespace Eigen::FFTOption;
    Eigen::FFT<Options> fft;
    if constexpr (CallSpec::Is1D) {
      fft.template fwd<NFFT0>(dst, src);
    } else {
      fft.template fwd<NFFT0, NFFT1>(dst, src);
    }
    TestAgainstOracle(dst, src, src.rows(), src.cols());
  }

  virtual void TestUnaryTransform() override final {
    using namespace Eigen::FFTOption;
    Eigen::FFT<Options> fft;
    dst = fft.fwd(src);
    TestAgainstOracle(dst, src, src.rows(), src.cols());
  }

  virtual void TestUnaryTransform(Eigen::Index nfft0, Eigen::Index nfft1) override final {
    using namespace Eigen::FFTOption;
    Eigen::FFT<Options> fft;
    if constexpr (CallSpec::Is1D) {
      dst = fft.fwd(src, nfft0);
    } else {
      dst = fft.fwd(src, nfft0, nfft1);
    }
    TestAgainstOracle(dst, src, src.rows(), src.cols());
  }

  virtual void TestUnaryTransformExplicitCompileTimeNFFT() override final {
    using namespace Eigen::FFTOption;
    Eigen::FFT<Options> fft;
    if constexpr (CallSpec::Is1D) {
      dst = fft.template fwd<NFFT0>(src);
    } else {
      dst = fft.template fwd<NFFT0, NFFT1>(src);
    }
    TestAgainstOracle(dst, src, src.rows(), src.cols());
  }

  virtual void TestUnaryTransformExplicitDstType() override final {
    using namespace Eigen::FFTOption;
    Eigen::FFT<Options> fft;
    dst = fft.template fwd<DstMatrixType>(src);
    TestAgainstOracle(dst, src, src.rows(), src.cols());
  }

  virtual void TestAgainstOracle(DstMatrixType& dst, SrcMatrixType& src, size_t nfft0, size_t nfft1) override final {
    {
      using namespace Eigen::FFTOption;

      // Handle resize in R2C case
      // If DstMatrixType is statically sized, we cannot resize to halfspectrum, hence cast to dynamic equivalent
      using DynamicDstType = std::conditional_t<
          CallSpec::Is1D,
          std::conditional_t<DstMatrixType::RowsAtCompileTime == 1, Eigen::RowVectorX<typename DstMatrixType::Scalar>,
                             Eigen::VectorX<typename DstMatrixType::Scalar>>,
          Eigen::MatrixX<typename DstMatrixType::Scalar>>;
      DynamicDstType oracle = Base::GetOracle(src, nfft0, nfft1).template cast<typename DstMatrixType::Scalar>();

      if constexpr (Options & HalfSpectrum) {
        if constexpr (CallSpec::Is1D) {
          oracle.conservativeResize(src.size() / 2 + 1);
        } else {
          oracle.conservativeResize(src.rows() / 2 + 1, src.cols());
        }
      }

      // No scaling of FFT(x) atm
      dst.normalize();
      oracle.normalize();

      // Conditions under which reduced array sizes can happen
      if constexpr (Options & HalfSpectrum) {
        if constexpr (CallSpec::Is1D) {
          ASSERT_EQ(dst.size(), src.size() / 2 + 1);
        } else {
          ASSERT_EQ(dst.rows(), src.rows() / 2 + 1);
        }
      }

      ASSERT_TRUE(dst.isApprox(oracle));
    }
  }
};

template <typename CallSpec>
class C2RTest : public FFTTestBase<CallSpec> {
  using Base = FFTTestBase<CallSpec>;
  using Base::dst;
  using Base::Options;
  using Base::src;
  using typename Base::DstMatrixType;
  using typename Base::SrcMatrixType;

 protected:
  C2RTest() : FFTTestBase<CallSpec>() { static_assert(CallSpec::RealTransform && CallSpec::Inverse); }

  // Assumes that src_d already has shape of FFT (i.e. fully allocated)
  template <typename DynamicSrcType>
  void ReflectSpectrum(DynamicSrcType& src_d) {
    const Eigen::Index rows = src_d.rows();
    const Eigen::Index cols = src_d.cols();
    for (Eigen::Index i = rows / 2 + 1; i < rows; i++) {
      for (Eigen::Index j = 0; j < cols; j++) {
        // Bottom half gets the conjugate of the corresponding top half element
        src_d(i, j) = std::conj(src_d(rows - i, (cols - j) % cols));
      }
    }
  }

  virtual void GenerateTestdata() override final {
    using namespace Eigen::FFTOption;
    if constexpr (CallSpec::Is1D) {
      if constexpr (CallSpec::SrcIsDynamic) {
        if constexpr (Options & HalfSpectrum) {
          src = SrcMatrixType::Random(NFFT0 / 2 + 1);
        } else {
          src = SrcMatrixType::Random(NFFT0);
        }
      } else {
        src = SrcMatrixType::Random();
      }
    } else {
      if constexpr (CallSpec::SrcIsDynamic) {
        if constexpr (Options & HalfSpectrum) {
          src = SrcMatrixType::Random(NFFT0 / 2 + 1, NFFT1);
        } else {
          src = SrcMatrixType::Random(NFFT0, NFFT1);
        }
      } else {
        src = SrcMatrixType::Random();
      }
    }
  }

  virtual void TestBinaryTransform() override final {
    using namespace Eigen::FFTOption;
    Eigen::FFT<Options> fft;

    // For binary call with no nfft arg (neither template nor runtime),
    // dst needs to be preallocated to avoid ambiguity of FFT shape
    if constexpr (CallSpec::DstIsDynamic) {
      if constexpr (CallSpec::Is1D) {
        dst.resize(NFFT0);
      } else {
        dst.resize(NFFT0, NFFT1);
      }
    }
    fft.inv(dst, src);
    TestAgainstOracle(dst, src, dst.rows(), dst.cols());  // src rows may be nfft0/2 + 1, hence use dst
  }

  virtual void TestBinaryTransform(Eigen::Index nfft0, Eigen::Index nfft1) override final {
    using namespace Eigen::FFTOption;
    Eigen::FFT<Options> fft;
    if constexpr (CallSpec::Is1D) {
      fft.inv(dst, src, nfft0);
    } else {
      fft.inv(dst, src, nfft0, nfft1);
    }
    TestAgainstOracle(dst, src, dst.rows(), dst.cols());  // src rows may be nfft0/2 + 1, hence use dst
  }

  virtual void TestBinaryTransformExplicitCompileTimeNFFT() override final {
    using namespace Eigen::FFTOption;
    Eigen::FFT<Options> fft;

    if constexpr (CallSpec::Is1D) {
      // fft.template inv<DstMatrixType, SrcMatrixType, NFFT0>(dst, src);
      fft.template inv<NFFT0>(dst, src);
    } else {
      // fft.template inv<DstMatrixType, SrcMatrixType, NFFT0, NFFT1>(dst, src);
      fft.template inv<NFFT0, NFFT1>(dst, src);
    }
    TestAgainstOracle(dst, src, dst.rows(), dst.cols());  // src rows may be nfft0/2 + 1, hence use dst
  }

  virtual void TestUnaryTransform() override final {
    using namespace Eigen::FFTOption;
    Eigen::FFT<Options> fft;

    if constexpr (CallSpec::Options & HalfSpectrum) {
      // Always need to specify nfft0 for C2R, as it is not possible to infer it from src
      if constexpr (CallSpec::Is1D) {
        dst = fft.inv(src, NFFT0);
      } else {
        dst = fft.inv(src, NFFT0, NFFT1);
      }
    } else {
      dst = fft.inv(src);
    }
    TestAgainstOracle(dst, src, dst.rows(), dst.cols());  // src rows may be nfft0/2 + 1, hence use dst
  }

  virtual void TestUnaryTransform(Eigen::Index nfft0, Eigen::Index nfft1) override final {
    using namespace Eigen::FFTOption;
    Eigen::FFT<Options> fft;
    if constexpr (CallSpec::Is1D) {
      dst = fft.inv(src, nfft0);
    } else {
      dst = fft.inv(src, nfft0, nfft1);
    }
    TestAgainstOracle(dst, src, dst.rows(), dst.cols());  // src rows may be nfft0/2 + 1, hence use dst
  }

  virtual void TestUnaryTransformExplicitCompileTimeNFFT() override final {
    using namespace Eigen::FFTOption;
    Eigen::FFT<Options> fft;

    if constexpr (CallSpec::Is1D) {
      dst = fft.template inv<NFFT0>(src);
    } else {
      dst = fft.template inv<NFFT0, NFFT1>(src);
    }
    TestAgainstOracle(dst, src, dst.rows(), dst.cols());  // src rows may be nfft0/2 + 1, hence use dst
  }

  virtual void TestUnaryTransformExplicitDstType() override final {
    using namespace Eigen::FFTOption;
    Eigen::FFT<Options> fft;
    if constexpr (CallSpec::DstIsDynamic) {
      if constexpr (CallSpec::Is1D) {
        dst = fft.template inv<DstMatrixType, NFFT0>(src);
      } else {
        dst = fft.template inv<DstMatrixType, NFFT0, NFFT1>(src);
      }
    } else {
      dst = fft.template inv<DstMatrixType>(src);
    }
    TestAgainstOracle(dst, src, dst.rows(), dst.cols());
  }

  virtual void TestAgainstOracle(DstMatrixType& dst, SrcMatrixType& src, size_t nfft0, size_t nfft1) override final {
    {
      using namespace Eigen::FFTOption;

      // Handle HalfSpectrum reflection in C2R case
      // If SrcMatrixType is statically sized, we cannot reflect to full spectrum, hence cast to dynamic equivalent
      using DynamicSrcType = std::conditional_t<CallSpec::Is1D, Eigen::VectorX<typename SrcMatrixType::Scalar>,
                                                Eigen::MatrixX<typename SrcMatrixType::Scalar>>;
      DynamicSrcType src_d = src;
      if constexpr (Options & HalfSpectrum) {
        if constexpr (CallSpec::Is1D) {
          ASSERT_EQ(dst.size() / 2 + 1, src.size());
          src_d.conservativeResize(dst.size());
        } else {
          ASSERT_EQ(dst.rows() / 2 + 1, src.rows());
          src_d.conservativeResize(dst.rows(), dst.cols());
        }
      }
      ReflectSpectrum(src_d);
      DstMatrixType oracle =
          Base::GetOracle(src_d, nfft0, nfft1).template cast<typename SrcMatrixType::Scalar>().real();

      // IFFT(FFT(x)) == a * x if Unscaled is enabled
      if constexpr (Options & Unscaled) {
        dst.normalize();
        oracle.normalize();
      } else {
        // Mimic scaling done by Eigen::FFT
        oracle /= oracle.size();
      }
      ASSERT_TRUE(dst.isApprox(oracle));
    }
  }
};