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

using C2CTransforms = ::testing::Types<
    FFTCallSpec<Eigen::VectorXcd, Eigen::VectorXcd, true, Options1>,
    FFTCallSpec<Eigen::MatrixXcd, Eigen::MatrixXcd, true, Options1>,
    FFTCallSpec<Eigen::Vector3cd, Eigen::Vector3cd, true, Options1>,
    FFTCallSpec<Eigen::Matrix3cd, Eigen::Matrix3cd, true, Options1>,

    FFTCallSpec<Eigen::VectorXcd, Eigen::VectorXcd, true, Options2>,
    FFTCallSpec<Eigen::MatrixXcd, Eigen::MatrixXcd, true, Options2>,
    FFTCallSpec<Eigen::Vector3cd, Eigen::Vector3cd, true, Options2>,
    FFTCallSpec<Eigen::Matrix3cd, Eigen::Matrix3cd, true, Options2>,
    
    FFTCallSpec<Eigen::VectorXcd, Eigen::VectorXcd, true, Options3>,
    FFTCallSpec<Eigen::MatrixXcd, Eigen::MatrixXcd, true, Options3>,
    FFTCallSpec<Eigen::Vector3cd, Eigen::Vector3cd, true, Options3>,
    FFTCallSpec<Eigen::Matrix3cd, Eigen::Matrix3cd, true, Options3>,

    FFTCallSpec<Eigen::VectorXcd, Eigen::VectorXcd, true, Options4>,
    FFTCallSpec<Eigen::MatrixXcd, Eigen::MatrixXcd, true, Options4>,
    FFTCallSpec<Eigen::Vector3cd, Eigen::Vector3cd, true, Options4>,
    FFTCallSpec<Eigen::Matrix3cd, Eigen::Matrix3cd, true, Options4>,
    
    FFTCallSpec<Eigen::VectorXcd, Eigen::VectorXcd, true, Options5>,
    FFTCallSpec<Eigen::MatrixXcd, Eigen::MatrixXcd, true, Options5>,
    FFTCallSpec<Eigen::Vector3cd, Eigen::Vector3cd, true, Options5>,
    FFTCallSpec<Eigen::Matrix3cd, Eigen::Matrix3cd, true, Options5>,

    FFTCallSpec<Eigen::VectorXcd, Eigen::VectorXcd, true, Options6>,
    FFTCallSpec<Eigen::MatrixXcd, Eigen::MatrixXcd, true, Options6>,
    FFTCallSpec<Eigen::Vector3cd, Eigen::Vector3cd, true, Options6>,
    FFTCallSpec<Eigen::Matrix3cd, Eigen::Matrix3cd, true, Options6>,

    FFTCallSpec<Eigen::VectorXcd, Eigen::VectorXcd, true, Options7>,
    FFTCallSpec<Eigen::MatrixXcd, Eigen::MatrixXcd, true, Options7>,
    FFTCallSpec<Eigen::Vector3cd, Eigen::Vector3cd, true, Options7>,
    FFTCallSpec<Eigen::Matrix3cd, Eigen::Matrix3cd, true, Options7>,

    FFTCallSpec<Eigen::VectorXcd, Eigen::VectorXcd, true, Options8>,
    FFTCallSpec<Eigen::MatrixXcd, Eigen::MatrixXcd, true, Options8>,
    FFTCallSpec<Eigen::Vector3cd, Eigen::Vector3cd, true, Options8>,
    FFTCallSpec<Eigen::Matrix3cd, Eigen::Matrix3cd, true, Options8>
    >;

TYPED_TEST_SUITE(FFTTest, C2CTransforms);
TYPED_TEST(FFTTest, BinaryC2C) {
  std::cout << TypeParam::name();
  this->GenerateTestdata();
  this->TestBinaryTransform();
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
