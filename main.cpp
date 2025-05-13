#include <gtest/gtest.h>

#include "FFTTest.hpp"

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
using EigenStaticMatcd = Eigen::Matrix<std::complex<double>, NFFT0, NFFT1>;
using EigenStaticHalfSpectrumMatcd = Eigen::Matrix<std::complex<double>, NFFT0 / 2 + 1, NFFT1>;

using EigenStaticVecd = Eigen::Vector<double, NFFT0>;
using EigenStaticMatd = Eigen::Matrix<double, NFFT0, NFFT1>;

using C2COptionTests = ::testing::Types<FFTCallSpec<Eigen::VectorXcd, Eigen::VectorXcd, true, Options1>,
                                        FFTCallSpec<Eigen::MatrixXcd, Eigen::MatrixXcd, true, Options1>,
                                        FFTCallSpec<EigenStaticVeccd, EigenStaticVeccd, true, Options1>,
                                        FFTCallSpec<EigenStaticMatcd, EigenStaticMatcd, true, Options1>,

                                        FFTCallSpec<Eigen::VectorXcd, Eigen::VectorXcd, true, Options2>,
                                        FFTCallSpec<Eigen::MatrixXcd, Eigen::MatrixXcd, true, Options2>,
                                        FFTCallSpec<EigenStaticVeccd, EigenStaticVeccd, true, Options2>,
                                        FFTCallSpec<EigenStaticMatcd, EigenStaticMatcd, true, Options2>,

                                        FFTCallSpec<Eigen::VectorXcd, Eigen::VectorXcd, true, Options3>,
                                        FFTCallSpec<Eigen::MatrixXcd, Eigen::MatrixXcd, true, Options3>,
                                        FFTCallSpec<EigenStaticVeccd, EigenStaticVeccd, true, Options3>,
                                        FFTCallSpec<EigenStaticMatcd, EigenStaticMatcd, true, Options3>,

                                        FFTCallSpec<Eigen::VectorXcd, Eigen::VectorXcd, true, Options4>,
                                        FFTCallSpec<Eigen::MatrixXcd, Eigen::MatrixXcd, true, Options4>,
                                        FFTCallSpec<EigenStaticVeccd, EigenStaticVeccd, true, Options4>,
                                        FFTCallSpec<EigenStaticMatcd, EigenStaticMatcd, true, Options4>,

                                        FFTCallSpec<Eigen::VectorXcd, Eigen::VectorXcd, true, Options5>,
                                        FFTCallSpec<Eigen::MatrixXcd, Eigen::MatrixXcd, true, Options5>,
                                        FFTCallSpec<EigenStaticVeccd, EigenStaticVeccd, true, Options5>,
                                        FFTCallSpec<EigenStaticMatcd, EigenStaticMatcd, true, Options5>,

                                        FFTCallSpec<Eigen::VectorXcd, Eigen::VectorXcd, true, Options6>,
                                        FFTCallSpec<Eigen::MatrixXcd, Eigen::MatrixXcd, true, Options6>,
                                        FFTCallSpec<EigenStaticVeccd, EigenStaticVeccd, true, Options6>,
                                        FFTCallSpec<EigenStaticMatcd, EigenStaticMatcd, true, Options6>,

                                        FFTCallSpec<Eigen::VectorXcd, Eigen::VectorXcd, true, Options7>,
                                        FFTCallSpec<Eigen::MatrixXcd, Eigen::MatrixXcd, true, Options7>,
                                        FFTCallSpec<EigenStaticVeccd, EigenStaticVeccd, true, Options7>,
                                        FFTCallSpec<EigenStaticMatcd, EigenStaticMatcd, true, Options7>,

                                        FFTCallSpec<Eigen::VectorXcd, Eigen::VectorXcd, true, Options8>,
                                        FFTCallSpec<Eigen::MatrixXcd, Eigen::MatrixXcd, true, Options8>,
                                        FFTCallSpec<EigenStaticVeccd, EigenStaticVeccd, true, Options8>,
                                        FFTCallSpec<EigenStaticMatcd, EigenStaticMatcd, true, Options8>,

                                        FFTCallSpec<Eigen::VectorXcd, Eigen::VectorXcd, false, Options1>,
                                        FFTCallSpec<Eigen::MatrixXcd, Eigen::MatrixXcd, false, Options1>,
                                        FFTCallSpec<EigenStaticVeccd, EigenStaticVeccd, false, Options1>,
                                        FFTCallSpec<EigenStaticMatcd, EigenStaticMatcd, false, Options1>,

                                        FFTCallSpec<Eigen::VectorXcd, Eigen::VectorXcd, false, Options2>,
                                        FFTCallSpec<Eigen::MatrixXcd, Eigen::MatrixXcd, false, Options2>,
                                        FFTCallSpec<EigenStaticVeccd, EigenStaticVeccd, false, Options2>,
                                        FFTCallSpec<EigenStaticMatcd, EigenStaticMatcd, false, Options2>,

                                        FFTCallSpec<Eigen::VectorXcd, Eigen::VectorXcd, false, Options3>,
                                        FFTCallSpec<Eigen::MatrixXcd, Eigen::MatrixXcd, false, Options3>,
                                        FFTCallSpec<EigenStaticVeccd, EigenStaticVeccd, false, Options3>,
                                        FFTCallSpec<EigenStaticMatcd, EigenStaticMatcd, false, Options3>,

                                        FFTCallSpec<Eigen::VectorXcd, Eigen::VectorXcd, false, Options4>,
                                        FFTCallSpec<Eigen::MatrixXcd, Eigen::MatrixXcd, false, Options4>,
                                        FFTCallSpec<EigenStaticVeccd, EigenStaticVeccd, false, Options4>,
                                        FFTCallSpec<EigenStaticMatcd, EigenStaticMatcd, false, Options4>,

                                        FFTCallSpec<Eigen::VectorXcd, Eigen::VectorXcd, false, Options5>,
                                        FFTCallSpec<Eigen::MatrixXcd, Eigen::MatrixXcd, false, Options5>,
                                        FFTCallSpec<EigenStaticVeccd, EigenStaticVeccd, false, Options5>,
                                        FFTCallSpec<EigenStaticMatcd, EigenStaticMatcd, false, Options5>,

                                        FFTCallSpec<Eigen::VectorXcd, Eigen::VectorXcd, false, Options6>,
                                        FFTCallSpec<Eigen::MatrixXcd, Eigen::MatrixXcd, false, Options6>,
                                        FFTCallSpec<EigenStaticVeccd, EigenStaticVeccd, false, Options6>,
                                        FFTCallSpec<EigenStaticMatcd, EigenStaticMatcd, false, Options6>,

                                        FFTCallSpec<Eigen::VectorXcd, Eigen::VectorXcd, false, Options7>,
                                        FFTCallSpec<Eigen::MatrixXcd, Eigen::MatrixXcd, false, Options7>,
                                        FFTCallSpec<EigenStaticVeccd, EigenStaticVeccd, false, Options7>,
                                        FFTCallSpec<EigenStaticMatcd, EigenStaticMatcd, false, Options7>,

                                        FFTCallSpec<Eigen::VectorXcd, Eigen::VectorXcd, false, Options8>,
                                        FFTCallSpec<Eigen::MatrixXcd, Eigen::MatrixXcd, false, Options8>,
                                        FFTCallSpec<EigenStaticVeccd, EigenStaticVeccd, false, Options8>,
                                        FFTCallSpec<EigenStaticMatcd, EigenStaticMatcd, false, Options8> >;

using R2COptionTest = ::testing::Types<FFTCallSpec<Eigen::VectorXcd, Eigen::VectorXd, true, Options1>,
                                       FFTCallSpec<Eigen::MatrixXcd, Eigen::MatrixXd, true, Options1>,
                                       FFTCallSpec<EigenStaticVeccd, EigenStaticVecd, true, Options1>,
                                       FFTCallSpec<EigenStaticMatcd, EigenStaticMatd, true, Options1>,

                                       FFTCallSpec<Eigen::VectorXcd, Eigen::VectorXd, true, Options2>,
                                       FFTCallSpec<Eigen::MatrixXcd, Eigen::MatrixXd, true, Options2>,
                                       FFTCallSpec<EigenStaticVeccd, EigenStaticVecd, true, Options2>,
                                       FFTCallSpec<EigenStaticMatcd, EigenStaticMatd, true, Options2>,

                                       FFTCallSpec<Eigen::VectorXcd, Eigen::VectorXd, true, Options3>,
                                       FFTCallSpec<Eigen::MatrixXcd, Eigen::MatrixXd, true, Options3>,
                                       FFTCallSpec<EigenStaticVeccd, EigenStaticVecd, true, Options3>,
                                       FFTCallSpec<EigenStaticMatcd, EigenStaticMatd, true, Options3>,

                                       FFTCallSpec<Eigen::VectorXcd, Eigen::VectorXd, true, Options4>,
                                       FFTCallSpec<Eigen::MatrixXcd, Eigen::MatrixXd, true, Options4>,
                                       FFTCallSpec<EigenStaticVeccd, EigenStaticVecd, true, Options4>,
                                       FFTCallSpec<EigenStaticMatcd, EigenStaticMatd, true, Options4>,

                                       FFTCallSpec<Eigen::VectorXcd, Eigen::VectorXd, true, Options5>,
                                       FFTCallSpec<Eigen::MatrixXcd, Eigen::MatrixXd, true, Options5>,
                                       FFTCallSpec<EigenStaticHalfSpectrumVeccd, EigenStaticVecd, true, Options5>,
                                       FFTCallSpec<EigenStaticHalfSpectrumMatcd, EigenStaticMatd, true, Options5>,

                                       FFTCallSpec<Eigen::VectorXcd, Eigen::VectorXd, true, Options6>,
                                       FFTCallSpec<Eigen::MatrixXcd, Eigen::MatrixXd, true, Options6>,
                                       FFTCallSpec<EigenStaticHalfSpectrumVeccd, EigenStaticVecd, true, Options6>,
                                       FFTCallSpec<EigenStaticHalfSpectrumMatcd, EigenStaticMatd, true, Options6>,

                                       FFTCallSpec<Eigen::VectorXcd, Eigen::VectorXd, true, Options7>,
                                       FFTCallSpec<Eigen::MatrixXcd, Eigen::MatrixXd, true, Options7>,
                                       FFTCallSpec<EigenStaticHalfSpectrumVeccd, EigenStaticVecd, true, Options7>,
                                       FFTCallSpec<EigenStaticHalfSpectrumMatcd, EigenStaticMatd, true, Options7>,

                                       FFTCallSpec<Eigen::VectorXcd, Eigen::VectorXd, true, Options8>,
                                       FFTCallSpec<Eigen::MatrixXcd, Eigen::MatrixXd, true, Options8>,
                                       FFTCallSpec<EigenStaticHalfSpectrumVeccd, EigenStaticVecd, true, Options8>,
                                       FFTCallSpec<EigenStaticHalfSpectrumMatcd, EigenStaticMatd, true, Options8> >;

using C2ROptionTest = ::testing::Types<FFTCallSpec<Eigen::VectorXd, Eigen::VectorXcd, false, Options1>,
                                       FFTCallSpec<Eigen::MatrixXd, Eigen::MatrixXcd, false, Options1>,
                                       FFTCallSpec<EigenStaticVecd, EigenStaticVeccd, false, Options1>,
                                       FFTCallSpec<EigenStaticMatd, EigenStaticMatcd, false, Options1>,

                                       FFTCallSpec<Eigen::VectorXd, Eigen::VectorXcd, false, Options2>,
                                       FFTCallSpec<Eigen::MatrixXd, Eigen::MatrixXcd, false, Options2>,
                                       FFTCallSpec<EigenStaticVecd, EigenStaticVeccd, false, Options2>,
                                       FFTCallSpec<EigenStaticMatd, EigenStaticMatcd, false, Options2>,

                                       FFTCallSpec<Eigen::VectorXd, Eigen::VectorXcd, false, Options3>,
                                       FFTCallSpec<Eigen::MatrixXd, Eigen::MatrixXcd, false, Options3>,
                                       FFTCallSpec<EigenStaticVecd, EigenStaticVeccd, false, Options3>,
                                       FFTCallSpec<EigenStaticMatd, EigenStaticMatcd, false, Options3>,

                                       FFTCallSpec<Eigen::VectorXd, Eigen::VectorXcd, false, Options4>,
                                       FFTCallSpec<Eigen::MatrixXd, Eigen::MatrixXcd, false, Options4>,
                                       FFTCallSpec<EigenStaticVecd, EigenStaticVeccd, false, Options4>,
                                       FFTCallSpec<EigenStaticMatd, EigenStaticMatcd, false, Options4>,

                                       FFTCallSpec<Eigen::VectorXd, Eigen::VectorXcd, false, Options5>,
                                       FFTCallSpec<Eigen::MatrixXd, Eigen::MatrixXcd, false, Options5>,
                                       FFTCallSpec<EigenStaticVecd, EigenStaticHalfSpectrumVeccd, false, Options5>,
                                       FFTCallSpec<EigenStaticMatd, EigenStaticHalfSpectrumMatcd, false, Options5>,

                                       FFTCallSpec<Eigen::VectorXd, Eigen::VectorXcd, false, Options6>,
                                       FFTCallSpec<Eigen::MatrixXd, Eigen::MatrixXcd, false, Options6>,
                                       FFTCallSpec<EigenStaticVecd, EigenStaticHalfSpectrumVeccd, false, Options6>,
                                       FFTCallSpec<EigenStaticMatd, EigenStaticHalfSpectrumMatcd, false, Options6>,

                                       FFTCallSpec<Eigen::VectorXd, Eigen::VectorXcd, false, Options7>,
                                       FFTCallSpec<Eigen::MatrixXd, Eigen::MatrixXcd, false, Options7>,
                                       FFTCallSpec<EigenStaticVecd, EigenStaticHalfSpectrumVeccd, false, Options7>,
                                       FFTCallSpec<EigenStaticMatd, EigenStaticHalfSpectrumMatcd, false, Options7>,

                                       FFTCallSpec<Eigen::VectorXd, Eigen::VectorXcd, false, Options8>,
                                       FFTCallSpec<Eigen::MatrixXd, Eigen::MatrixXcd, false, Options8>,
                                       FFTCallSpec<EigenStaticVecd, EigenStaticHalfSpectrumVeccd, false, Options8>,
                                       FFTCallSpec<EigenStaticMatd, EigenStaticHalfSpectrumMatcd, false, Options8> >;

TYPED_TEST_SUITE(C2CTest, C2COptionTests);
TYPED_TEST(C2CTest, BinaryCalls) {
  std::cout << TypeParam::name();
  this->GenerateTestdata();
  this->TestBinaryTransform();
}

TYPED_TEST_SUITE(R2CTest, R2COptionTest);
TYPED_TEST(R2CTest, BinaryCalls) {
  std::cout << TypeParam::name();
  this->GenerateTestdata();
  this->TestBinaryTransform();
}

TYPED_TEST_SUITE(C2RTest, C2ROptionTest);
TYPED_TEST(C2RTest, BinaryCalls) {
  std::cout << TypeParam::name();
  this->GenerateTestdata();
  this->TestBinaryTransform();
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
