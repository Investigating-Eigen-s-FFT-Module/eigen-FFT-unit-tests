#pragma once
#include <Eigen/Dense>
#include <type_traits>

template <typename DstType, typename SrcType, bool Direction, int Opts>
struct FFTCallSpec {
  using SrcMatrixType = SrcType;
  using DstMatrixType = DstType;
  using Scalar = typename SrcType::Scalar;
  using RealScalar = typename SrcType::RealScalar;

  static_assert(std::is_same<RealScalar, typename DstType::RealScalar>::value &&
                "MISMATCHED SCALAR TPYE IN FTTCallSpec");
  static constexpr int Options = Opts;
  static constexpr bool SrcReal = std::is_same<typename SrcType::Scalar, typename SrcType::RealScalar>::value;
  static constexpr bool DstReal = std::is_same<typename DstType::Scalar, typename DstType::RealScalar>::value;
  static constexpr bool RealTransform = SrcReal || DstReal;
  static constexpr bool Forward = Direction;
  static constexpr bool Inverse = !Direction;
  static constexpr bool Is1D = SrcMatrixType::IsVectorAtCompileTime;
  static constexpr bool SrcIsDynamic = SrcMatrixType::RowsAtCompileTime == -1 || SrcMatrixType::RowsAtCompileTime == -1;
  static constexpr bool DstIsDynamic = DstMatrixType::RowsAtCompileTime == -1 || DstMatrixType::RowsAtCompileTime == -1;

  // static_assert((!SrcReal && !DstReal) || (SrcReal && !DstReal && Forward) || (!SrcReal && DstReal && Inverse));

  static std::string name() {
    std::stringstream ss;

    // Format matrix dimensions
    auto formatMatrix = [](const auto& type) -> std::string {
      using Type = std::decay_t<decltype(type)>;
      std::stringstream s;

      // Get value type
      using ValueType = typename Type::Scalar;
      if constexpr (std::is_same_v<ValueType, std::complex<float>>)
        s << "complex<float>";
      else if constexpr (std::is_same_v<ValueType, std::complex<double>>)
        s << "complex<double>";
      else if constexpr (std::is_same_v<ValueType, float>)
        s << "float";
      else if constexpr (std::is_same_v<ValueType, double>)
        s << "double";
      else
        s << "unknown";

      s << " Matrix(";

      // Get dimensions
      if constexpr (Type::RowsAtCompileTime == -1)
        s << "Dynamic";
      else
        s << Type::RowsAtCompileTime;

      s << "x";

      if constexpr (Type::ColsAtCompileTime == -1)
        s << "Dynamic";
      else
        s << Type::ColsAtCompileTime;

      s << ")";
      return s.str();
    };

    ss << "Src: " << formatMatrix(SrcType()) << std::endl
       << "Dst: " << formatMatrix(DstType()) << std::endl
       << (Forward ? "Forward" : "Inverse") << std::endl
       << (SrcReal ? "R" : "C") << "2" << (DstReal ? "R" : "C") << std::endl
       << "Options: " << (Opts & Eigen::FFTOption::Scaled ? "Scaled" : "Unscaled") << ", "
       << (Opts & Eigen::FFTOption::HalfSpectrum ? "HalfSpectrum" : "FullSpectrum") << std::endl;

    return ss.str();
  }
};