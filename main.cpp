#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>
#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>
#include <vector>
#include <random>
#include <complex>
#include <cmath>

#include "data_gen.hpp"
#include "pocketfft_hdronly.h"

using big_float_t = long double;
using big_complex_t = std::complex<big_float_t>;

// complex input to oracle
template <typename SCALAR_T>
Eigen::MatrixX<std::complex<SCALAR_T>> oracle_fwd(const Eigen::MatrixX<std::complex<SCALAR_T>>& src) {
    Eigen::MatrixX<big_complex_t> src_cp = src.template cast<big_complex_t>();
    big_complex_t* src_ptr = src_cp.data();

    using namespace pocketfft;

    const size_t nfft0 = src_cp.rows();
    const size_t nfft1 = src_cp.cols();
    const shape_t shape{nfft0, nfft1};
    const shape_t axes{0, 1};
    const stride_t stride{
        static_cast<ptrdiff_t>(sizeof(big_complex_t) * nfft1),
        static_cast<ptrdiff_t>(sizeof(big_complex_t))
    };

    Eigen::MatrixX<big_complex_t> dst(nfft0, nfft1);
    big_complex_t* dst_ptr = dst.data();

    c2c(shape, stride, stride, axes, FORWARD, src_ptr, dst_ptr, static_cast<big_float_t>(1), static_cast<size_t>(1));

    Eigen::MatrixX<std::complex<SCALAR_T>> dst_cp = dst.cast<std::complex<SCALAR_T>>();

    return dst_cp;
}

namespace utf = boost::unit_test;

// Main test suite
BOOST_AUTO_TEST_SUITE(EigenFFTTests)

BOOST_AUTO_TEST_CASE(test, * utf::tolerance(1e-15))
{
    Eigen::VectorXf mat(5);
    mat.setConstant(1.);

    Eigen::FFT<float> fft;

    Eigen::VectorXcf res = fft.fwd(mat);

    Eigen::VectorXcf res_oracle = oracle_fwd<float>(mat);

    BOOST_TEST(res_oracle == res);
}

BOOST_AUTO_TEST_SUITE_END()