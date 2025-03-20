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
using flag_t = int;
static const flag_t Default = 0;
static const flag_t Unscaled = 1;
static const flag_t HalfSpectrum = 2;
static const flag_t Speedy = 32767;

// src can be Eigen::MatrixX<SCALAR_T>; implicit conversion works for same scalar types
template <typename SCALAR_T>
Eigen::MatrixX<std::complex<SCALAR_T>> oracle_fwd(const Eigen::MatrixX<std::complex<SCALAR_T>>& src, flag_t flags=Default) {
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

    // no scaling on fwd atm, also why it's not sqrt factor.
    // if (!(flags & Unscaled)) {
    //     dst /= nfft0 * nfft1;
    // }

    Eigen::MatrixX<std::complex<SCALAR_T>> dst_cp = dst.cast<std::complex<SCALAR_T>>();

    return dst_cp;
}

template <typename T>
struct is_complex : std::false_type {};

template <typename T>
struct is_complex<std::complex<T>> : std::true_type {};

template <typename SCALAR_T, typename RETURN_T = SCALAR_T>
auto oracle_inv(const Eigen::MatrixX<std::complex<SCALAR_T>>& src, flag_t flags=Default) {
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

    c2c(shape, stride, stride, axes, BACKWARD, src_ptr, dst_ptr, static_cast<big_float_t>(1), static_cast<size_t>(1));

    if (!(flags & Unscaled)) {
        dst /= nfft0 * nfft1;
    }

    if constexpr (is_complex<RETURN_T>::value) {
        Eigen::MatrixX<std::complex<SCALAR_T>> dst_cp = dst.template cast<std::complex<SCALAR_T>>();
        return dst_cp;
    }
    else {
        Eigen::MatrixX<SCALAR_T> dst_cp = dst.real().template cast<SCALAR_T>();
        return dst_cp;
    }
}

namespace utf = boost::unit_test;

// Main test suite
BOOST_AUTO_TEST_SUITE(EigenFFTTests)

BOOST_AUTO_TEST_CASE(test_oracle_fwd, * utf::tolerance(1e-15))
{
    Eigen::VectorXcf mat(5);
    mat.setConstant(1.);

    Eigen::FFT<float> fft;
    
    Eigen::VectorXcf res(5);
    fft.fwd(res, mat);
    
    Eigen::VectorXcf res_oracle = oracle_fwd<float>(mat);
    
    BOOST_TEST(res_oracle == res);
}

BOOST_AUTO_TEST_CASE(test_oracle_inv, * utf::tolerance(1e-15))
{
    Eigen::VectorXcd mat(2);
    mat.setConstant({1., 1.});

    Eigen::FFT<double> fft;

    Eigen::VectorXcd res(2);
    fft.inv(res, mat, 2);

    Eigen::VectorXcd res_oracle = oracle_inv<double, std::complex<double>>(mat);

    BOOST_TEST(res_oracle == res);
}

BOOST_AUTO_TEST_SUITE_END()