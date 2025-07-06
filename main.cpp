#include <gtest/gtest.h>

#include "FFTTest.hpp"
#include "macros.h"

using C2COptionTests = ::testing::Types<C2C_OPTION_TESTS>;
using R2COptionTests = ::testing::Types<R2C_OPTION_TESTS>;
using C2ROptionTests = ::testing::Types<C2R_OPTION_TESTS>;

TYPED_TEST_SUITE(C2CTest, C2COptionTests);
TYPED_TEST(C2CTest, BinaryCalls) {
  std::cout << TypeParam::name();
  this->GenerateTestdata();
  this->TestBinaryTransform();
}
TYPED_TEST(C2CTest, UnaryCalls) {
  std::cout << TypeParam::name();
  this->GenerateTestdata();
  this->TestUnaryTransform();
}
TYPED_TEST(C2CTest, BinaryCallsWithRuntimeNFFT) {
  std::cout << TypeParam::name();
  this->GenerateTestdata();
  this->TestBinaryTransform(NFFT0, NFFT1);
}
TYPED_TEST(C2CTest, UnaryCallsWithRuntimeNFFT) {
  std::cout << TypeParam::name();
  this->GenerateTestdata();
  this->TestUnaryTransform(NFFT0, NFFT1);
}
TYPED_TEST(C2CTest, BinaryCallsWithCompiletimeNFFT) {
  std::cout << TypeParam::name();
  this->GenerateTestdata();
  this->TestBinaryTransformExplicitCompileTimeNFFT();
}
TYPED_TEST(C2CTest, UnaryCallsWithCompiletimeNFFT) {
  std::cout << TypeParam::name();
  this->GenerateTestdata();
  this->TestUnaryTransformExplicitCompileTimeNFFT();
}

TYPED_TEST_SUITE(R2CTest, R2COptionTests);
TYPED_TEST(R2CTest, BinaryCalls) {
  std::cout << TypeParam::name();
  this->GenerateTestdata();
  this->TestBinaryTransform();
}
TYPED_TEST(R2CTest, UnaryCalls) {
  std::cout << TypeParam::name();
  this->GenerateTestdata();
  this->TestUnaryTransform();
}
TYPED_TEST(R2CTest, BinaryCallsWithRuntimeNFFT) {
  std::cout << TypeParam::name();
  this->GenerateTestdata();
  this->TestBinaryTransform(NFFT0, NFFT1);
}
TYPED_TEST(R2CTest, UnaryCallsWithRuntimeNFFT) {
  std::cout << TypeParam::name();
  this->GenerateTestdata();
  this->TestUnaryTransform(NFFT0, NFFT1);
}
TYPED_TEST(R2CTest, BinaryCallsWithCompiletimeNFFT) {
  std::cout << TypeParam::name();
  this->GenerateTestdata();
  this->TestBinaryTransformExplicitCompileTimeNFFT();
}
TYPED_TEST(R2CTest, UnaryCallsWithCompiletimeNFFT) {
  std::cout << TypeParam::name();
  this->GenerateTestdata();
  this->TestUnaryTransformExplicitCompileTimeNFFT();
}

TYPED_TEST_SUITE(C2RTest, C2ROptionTests);
TYPED_TEST(C2RTest, BinaryCalls) {
  std::cout << TypeParam::name();
  this->GenerateTestdata();
  this->TestBinaryTransform();
}
TYPED_TEST(C2RTest, UnaryCalls) {
  std::cout << TypeParam::name();
  this->GenerateTestdata();
  this->TestUnaryTransform();
}
TYPED_TEST(C2RTest, BinaryCallsWithRuntimeNFFT) {
  std::cout << TypeParam::name();
  this->GenerateTestdata();
  this->TestBinaryTransform(NFFT0, NFFT1);
}
TYPED_TEST(C2RTest, UnaryCallsWithRuntimeNFFT) {
  std::cout << TypeParam::name();
  this->GenerateTestdata();
  this->TestUnaryTransform(NFFT0, NFFT1);
}
TYPED_TEST(C2RTest, BinaryCallsWithCompiletimeNFFT) {
  std::cout << TypeParam::name();
  this->GenerateTestdata();
  this->TestBinaryTransformExplicitCompileTimeNFFT();
}
TYPED_TEST(C2RTest, UnaryCallsWithCompiletimeNFFT) {
  std::cout << TypeParam::name();
  this->GenerateTestdata();
  this->TestUnaryTransformExplicitCompileTimeNFFT();
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
