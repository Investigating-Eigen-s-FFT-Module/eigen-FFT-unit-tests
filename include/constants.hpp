#pragma once

#include <Eigen/Dense>
#include <complex>
#include <unsupported/Eigen/FFT>

constexpr int NFFT0 = 7;
constexpr int NFFT1 = 4;

constexpr int Options1 = Eigen::FFTOption::Scaled | Eigen::FFTOption::OutPlace | Eigen::FFTOption::FullSpectrum;
constexpr int Options2 = Eigen::FFTOption::Unscaled | Eigen::FFTOption::OutPlace | Eigen::FFTOption::FullSpectrum;
constexpr int Options3 = Eigen::FFTOption::Scaled | Eigen::FFTOption::InPlace | Eigen::FFTOption::FullSpectrum;
constexpr int Options4 = Eigen::FFTOption::Unscaled | Eigen::FFTOption::InPlace | Eigen::FFTOption::FullSpectrum;
constexpr int Options5 = Eigen::FFTOption::Scaled | Eigen::FFTOption::OutPlace | Eigen::FFTOption::HalfSpectrum;
constexpr int Options6 = Eigen::FFTOption::Unscaled | Eigen::FFTOption::OutPlace | Eigen::FFTOption::HalfSpectrum;
constexpr int Options7 = Eigen::FFTOption::Scaled | Eigen::FFTOption::InPlace | Eigen::FFTOption::HalfSpectrum;
constexpr int Options8 = Eigen::FFTOption::Unscaled | Eigen::FFTOption::InPlace | Eigen::FFTOption::HalfSpectrum;

using EigenStaticVeccd = Eigen::Vector<std::complex<double>, NFFT0>;
using EigenStaticHalfSpectrumVeccd = Eigen::Vector<std::complex<double>, NFFT0 / 2 + 1>;
using EigenStaticVecd = Eigen::Vector<double, NFFT0>;

using EigenStaticMatcd = Eigen::Matrix<std::complex<double>, NFFT0, NFFT1>;
using EigenStaticHalfSpectrumMatcd = Eigen::Matrix<std::complex<double>, NFFT0 / 2 + 1, NFFT1>;
using EigenStaticMatd = Eigen::Matrix<double, NFFT0, NFFT1>;

using EigenStaticRowVeccd = Eigen::RowVector<std::complex<double>, NFFT0>;
using EigenStaticHalfSpectrumRowVeccd = Eigen::RowVector<std::complex<double>, NFFT0 / 2 + 1>;
using EigenStaticRowVecd = Eigen::RowVector<double, NFFT0>;