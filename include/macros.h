#pragma once

#include "constants.hpp"
#include "test_types.hpp"

#define GENERATE_TESTING_TYPES_ALL_OPTIONS(DSTTYPE, SRCTYPE, DIRECTION)                                   \
  FFTCallSpec<DSTTYPE, SRCTYPE, DIRECTION, Options1>, FFTCallSpec<DSTTYPE, SRCTYPE, DIRECTION, Options2>, \
      FFTCallSpec<DSTTYPE, SRCTYPE, DIRECTION, Options3>, FFTCallSpec<DSTTYPE, SRCTYPE, DIRECTION, Options4>

#define GENERATE_TESTING_TYPES_FULLSPECTRUM(DSTTYPE, SRCTYPE, DIRECTION) \
  FFTCallSpec<DSTTYPE, SRCTYPE, DIRECTION, Options1>, FFTCallSpec<DSTTYPE, SRCTYPE, DIRECTION, Options2>

#define GENERATE_TESTING_TYPES_HALFSPECTRUM(DSTTYPE, SRCTYPE, DIRECTION) \
  FFTCallSpec<DSTTYPE, SRCTYPE, DIRECTION, Options3>, FFTCallSpec<DSTTYPE, SRCTYPE, DIRECTION, Options4>

#define GENERATE_TESTING_TYPES_ALL_OPTIONS_C2C(DSTTYPE, SRCTYPE) \
  GENERATE_TESTING_TYPES_ALL_OPTIONS(DSTTYPE, SRCTYPE, true),    \
      GENERATE_TESTING_TYPES_ALL_OPTIONS(DSTTYPE, SRCTYPE, false)

#define GENERATE_TESTING_TYPES_ALL_OPTIONS_R2C(DSTTYPE, SRCTYPE, HALFSPECTRUM_DSTTYPE) \
  GENERATE_TESTING_TYPES_FULLSPECTRUM(DSTTYPE, SRCTYPE, true),                         \
      GENERATE_TESTING_TYPES_HALFSPECTRUM(HALFSPECTRUM_DSTTYPE, SRCTYPE, true)

#define GENERATE_TESTING_TYPES_ALL_OPTIONS_C2R(DSTTYPE, SRCTYPE, HALFSPECTRUM_SRCTYPE) \
  GENERATE_TESTING_TYPES_FULLSPECTRUM(DSTTYPE, SRCTYPE, false),                        \
      GENERATE_TESTING_TYPES_HALFSPECTRUM(DSTTYPE, HALFSPECTRUM_SRCTYPE, false)

#define C2C_OPTION_TESTS                                                          \
  GENERATE_TESTING_TYPES_ALL_OPTIONS_C2C(Eigen::MatrixXcd, Eigen::MatrixXcd),     \
      GENERATE_TESTING_TYPES_ALL_OPTIONS_C2C(Eigen::VectorXcd, Eigen::VectorXcd), \
      GENERATE_TESTING_TYPES_ALL_OPTIONS_C2C(EigenStaticVeccd, EigenStaticVeccd), \
      GENERATE_TESTING_TYPES_ALL_OPTIONS_C2C(EigenStaticMatcd, EigenStaticMatcd), \
      GENERATE_TESTING_TYPES_ALL_OPTIONS_C2C(EigenStaticMatcd, Eigen::MatrixXcd), \
      GENERATE_TESTING_TYPES_ALL_OPTIONS_C2C(EigenStaticVeccd, Eigen::VectorXcd), \
      GENERATE_TESTING_TYPES_ALL_OPTIONS_C2C(Eigen::MatrixXcd, EigenStaticMatcd), \
      GENERATE_TESTING_TYPES_ALL_OPTIONS_C2C(Eigen::VectorXcd, EigenStaticVeccd)

#define R2C_OPTION_TESTS                                                                                       \
  GENERATE_TESTING_TYPES_ALL_OPTIONS_R2C(Eigen::VectorXcd, Eigen::VectorXd, Eigen::VectorXcd),                 \
      GENERATE_TESTING_TYPES_ALL_OPTIONS_R2C(Eigen::MatrixXcd, Eigen::MatrixXd, Eigen::MatrixXcd),             \
      GENERATE_TESTING_TYPES_ALL_OPTIONS_R2C(EigenStaticVeccd, EigenStaticVecd, EigenStaticHalfSpectrumVeccd), \
      GENERATE_TESTING_TYPES_ALL_OPTIONS_R2C(EigenStaticMatcd, EigenStaticMatd, EigenStaticHalfSpectrumMatcd), \
      GENERATE_TESTING_TYPES_ALL_OPTIONS_R2C(EigenStaticMatcd, Eigen::MatrixXd, EigenStaticHalfSpectrumMatcd), \
      GENERATE_TESTING_TYPES_ALL_OPTIONS_R2C(EigenStaticVeccd, Eigen::VectorXd, EigenStaticHalfSpectrumVeccd), \
      GENERATE_TESTING_TYPES_ALL_OPTIONS_R2C(Eigen::MatrixXcd, EigenStaticMatd, Eigen::MatrixXcd),             \
      GENERATE_TESTING_TYPES_ALL_OPTIONS_R2C(Eigen::VectorXcd, EigenStaticVecd, Eigen::VectorXcd)

#define C2R_OPTION_TESTS                                                                                       \
  GENERATE_TESTING_TYPES_ALL_OPTIONS_C2R(Eigen::VectorXd, Eigen::VectorXcd, Eigen::VectorXcd),                 \
      GENERATE_TESTING_TYPES_ALL_OPTIONS_C2R(Eigen::MatrixXd, Eigen::MatrixXcd, Eigen::MatrixXcd),             \
      GENERATE_TESTING_TYPES_ALL_OPTIONS_C2R(EigenStaticVecd, EigenStaticVeccd, EigenStaticHalfSpectrumVeccd), \
      GENERATE_TESTING_TYPES_ALL_OPTIONS_C2R(EigenStaticMatd, EigenStaticMatcd, EigenStaticHalfSpectrumMatcd), \
      GENERATE_TESTING_TYPES_ALL_OPTIONS_C2R(EigenStaticMatd, Eigen::MatrixXcd, Eigen::MatrixXcd),             \
      GENERATE_TESTING_TYPES_ALL_OPTIONS_C2R(EigenStaticVecd, Eigen::VectorXcd, Eigen::VectorXcd),             \
      GENERATE_TESTING_TYPES_ALL_OPTIONS_C2R(Eigen::MatrixXd, EigenStaticMatcd, EigenStaticHalfSpectrumMatcd), \
      GENERATE_TESTING_TYPES_ALL_OPTIONS_C2R(Eigen::VectorXd, EigenStaticVeccd, EigenStaticHalfSpectrumVeccd)