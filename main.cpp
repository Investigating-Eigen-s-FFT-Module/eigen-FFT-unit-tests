#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>
#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>
#include <vector>
#include <random>
#include <complex>
#include <iomanip>


#include "data_gen.hpp"
#include "pocketfft_hdronly.h"

using big_float_t = long double;
using big_complex_t = std::complex<big_float_t>;
using flag_t = int;
static const flag_t Default = 0;
static const flag_t Unscaled = 1;
static const flag_t HalfSpectrum = 2;
static const flag_t Speedy = 32767;

constexpr inline bool hasFlag(flag_t flag, flag_t flags) {
    return static_cast<bool>(flags & flag);
}

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
    // if (!hasFlag(Unscaled, flags)) {
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

    if (!hasFlag(Unscaled, flags)) {
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

// Define tolerance for floating point comparisons based on precision
template <typename T>
T get_tolerance() {
    if constexpr (std::is_same_v<T, float>) {
        return 1e-5f;
    } else if constexpr (std::is_same_v<T, double>) {
        return 1e-10;
    } else {
        return static_cast<T>(1e-14); // long double
    }
}

// Configuration for test matrix sizes
struct TestDimensions {
    static const std::vector<std::pair<size_t, size_t>> getTestSizes() {
        return {
            {8, 8},
            {32, 32},
            {64, 64},
            {16, 8},
            {8, 16},
            {17, 19},
            {32, 1},
            {1, 32},
            {128, 128},
        };
    }
    
    static const std::vector<std::pair<size_t, size_t>> getSimpleSizes() {
        return {
            {8, 1},
            {16, 1},
            {17, 1}
        };
    }
};

// FFTTestBase flags 
static const flag_t Preallocate = 2;
static const flag_t UnaryFFT = 4;
static const flag_t KeepPlans = 8;
static const flag_t Real = 16;
static const flag_t InPlace = 32;
static const flag_t UseVectors = 64;

// TODO: IMPLEMENT MATRIX TEMPLATE FLAGS
template <typename SCALAR_T, flag_t FFT_FLAGS=Default, flag_t MATRIX_FLAGS=Default, flag_t TEST_FLAGS=Default> 
class FFTTest {
    public:
        using complex_t = std::complex<SCALAR_T>;

        template <typename FUNC_T>
        void test_roundtrip_func(const std::vector<std::pair<size_t, size_t>>& sizes, FUNC_T f) {
            Eigen::FFT<SCALAR_T> fft(Eigen::default_fft_impl<SCALAR_T>(), FFT_FLAGS);
            Eigen::MatrixX<SCALAR_T> input_real;
            Eigen::MatrixX<complex_t> input_complex;
            Eigen::MatrixX<SCALAR_T> output_real;
            Eigen::MatrixX<complex_t> output_complex;
            Eigen::MatrixX<complex_t> freq;
            Eigen::VectorX<SCALAR_T> input_real_v;
            Eigen::VectorX<complex_t> input_complex_v;
            Eigen::VectorX<SCALAR_T> output_real_v;
            Eigen::VectorX<complex_t> output_complex_v;
            Eigen::VectorX<complex_t> freq_v;

            for (const auto& [rows, cols] : sizes) {
                if constexpr (hasFlag(UseVectors, TEST_FLAGS)) {
                    BOOST_TEST_CONTEXT("Vector size: " << rows << get_flags_context()) {
                        // input generation
                        if constexpr (hasFlag(Real, TEST_FLAGS)) {
                            Eigen::MatrixX<std::pair<SCALAR_T, SCALAR_T>> coords = real_gen.generateCoordinateMatrix(rows, 1);
                            input_real_v.resize(rows);
                            for (size_t i = 0; i < rows; i++) {
                                auto& [x, y] = coords(i, 0);
                                input_real_v(i, 0) = f(x, y);
                            }
                        }
                        else {
                            input_complex_v = complex_gen.generateCoordinateMatrix(rows, 1);
                            input_complex_v.unaryExpr(f);
                        }

                        // Frequency vector
                        if constexpr (hasFlag(Preallocate, TEST_FLAGS)) {
                            if constexpr (hasFlag(HalfSpectrum, FFT_FLAGS)) {
                                freq = Eigen::VectorX<complex_t>(rows/2 + 1); // TODO: InPlace also needs padding.
                            }
                            else {
                                freq = Eigen::VectorX<complex_t>(rows);
                            }

                            if constexpr (hasFlag(Real, TEST_FLAGS)) {
                                output_real_v = Eigen::VectorX<SCALAR_T>(rows);
                            }
                            else {
                                output_complex_v = Eigen::VectorX<complex_t>(rows);
                            }
                        }

                        if constexpr (hasFlag(InPlace, TEST_FLAGS)) {
                            // TODO
                            BOOST_TEST_MESSAGE("InPlace is currently not supported yet.");
                            return;
                        }

                        // Actual round trip
                        if constexpr (hasFlag(UnaryFFT, TEST_FLAGS)) {
                            if constexpr (hasFlag(Real, TEST_FLAGS)) {
                                BOOST_TEST_MESSAGE("c2r using unary operator is currently not supported yet.");
                                return;
                                // freq = fft.fwd(input_real);
                                // output_real = fft.inv(freq);
                                // BOOST_CHECK_MESSAGE(input_real.isApprox(output_real, get_tolerance<SCALAR_T>()),
                                //                     "FFT->IFFT roundtrip failed");
                            }
                            else {
                                freq_v = fft.fwd(input_complex_v);
                                output_complex_v = fft.inv(freq_v);
                                BOOST_CHECK_MESSAGE(input_complex_v.isApprox(output_complex_v, get_tolerance<SCALAR_T>()),
                                                "FFT->IFFT roundtrip failed");
                            }
                        }
                        else {
                            if constexpr (hasFlag(Real, TEST_FLAGS)) {
                                fft.fwd(freq_v, input_real_v);
                                fft.inv(output_real_v, freq_v);
                                BOOST_CHECK_MESSAGE(input_real_v.isApprox(output_real_v, get_tolerance<SCALAR_T>()),
                                                "FFT->IFFT roundtrip failed");
                            }
                            else {
                                fft.fwd(freq_v, input_complex_v);
                                fft.inv(output_complex_v, freq_v);
                                BOOST_CHECK_MESSAGE(input_complex_v.isApprox(output_complex_v, get_tolerance<SCALAR_T>()),
                                                "FFT->IFFT roundtrip failed");
                            }
                        }
                    }
                }
                else {
                    BOOST_TEST_CONTEXT("Matrix size: " << rows << "x" << cols << get_flags_context()) {
                        // input generation
                        if constexpr (hasFlag(Real, TEST_FLAGS)) {
                            Eigen::MatrixX<std::pair<SCALAR_T, SCALAR_T>> coords = real_gen.generateCoordinateMatrix(rows, cols);
                            input_real.resize(rows, cols);
                            for (size_t i = 0; i < rows; i++) {
                                for (size_t j = 0; j < cols; j++) {
                                    auto& [x, y] = coords(i, j);
                                    input_real(i, j) = f(x, y);
                                }
                            }
                        }
                        else {
                            input_complex = complex_gen.generateCoordinateMatrix(rows, cols);
                            input_complex.unaryExpr(f);
                        }
    
                        // Frequency matrix
                        if constexpr (hasFlag(Preallocate, TEST_FLAGS)) {
                            if constexpr (hasFlag(HalfSpectrum, FFT_FLAGS)) {
                                freq = Eigen::MatrixX<complex_t>(rows/2 + 1, cols); // TODO: when allowing row-major order, this needs 2 cases
                                                                                    //       InPlace also needs padding.
                            }
                            else {
                                freq = Eigen::MatrixX<complex_t>(rows, cols);
                            }

                            if constexpr (hasFlag(Real, TEST_FLAGS)) {
                                output_real = Eigen::MatrixX<SCALAR_T>(rows, cols);
                            }
                            else {
                                output_complex = Eigen::MatrixX<complex_t>(rows, cols);
                            }
                        }
    
                        if constexpr (hasFlag(InPlace, TEST_FLAGS)) {
                            // TODO
                            BOOST_TEST_MESSAGE("InPlace is currently not supported yet.");
                            return;
                        }
    
                        // Actual round trip
                        if constexpr (hasFlag(UnaryFFT, TEST_FLAGS)) {
                            if constexpr (hasFlag(Real, TEST_FLAGS)) {
                                BOOST_TEST_MESSAGE("c2r using unary operator is currently not supported yet.");
                                return;
                                // freq = fft.fwd(input_real);
                                // output_real = fft.inv(freq);
                                // BOOST_CHECK_MESSAGE(input_real.isApprox(output_real, get_tolerance<SCALAR_T>()),
                                //                     "FFT->IFFT roundtrip failed");
                            }
                            else {
                                // TODO
                                BOOST_TEST_MESSAGE("Matrix input is currently not supported yet.");
                                return;
                                // freq = fft.fwd(input_complex);
                                // output_complex = fft.inv(freq);
                                // BOOST_CHECK_MESSAGE(input_complex.isApprox(output_complex, get_tolerance<SCALAR_T>()),
                                //                     "FFT->IFFT roundtrip failed");
                            }
                        }
                        else {
                            if constexpr (hasFlag(Real, TEST_FLAGS)) {
                                BOOST_TEST_MESSAGE("Matrix input is currently not supported yet.");
                                return;
                                // TODO: allow once matrix input is fine
                                // fft.fwd(freq, input_real);
                                // fft.inv(output_real, freq);
                                // if constexpr(hasFlag(Unscaled, FFT_FLAGS)) {
                                //     input_real.normalize();
                                //     output_real.normalize();
                                // }
                                // BOOST_CHECK_MESSAGE(input_real.isApprox(output_real, get_tolerance<SCALAR_T>()),
                                //                     "FFT->IFFT roundtrip failed");
                            }
                            else {
                                BOOST_TEST_MESSAGE("Matrix input is currently not supported yet.");
                                return;
                                // TODO: allow once matrix input is fine
                                // fft.fwd(freq, input_complex);
                                // fft.inv(output_complex, freq);
                                // if constexpr(hasFlag(Unscaled, FFT_FLAGS)) {
                                //     input_complex.normalize();
                                //     output_complex.normalize();
                                // }
                                // BOOST_CHECK_MESSAGE(input_complex.isApprox(output_complex, get_tolerance<SCALAR_T>()),
                                //                     "FFT->IFFT roundtrip failed");
                            }
                        }
                    }
                }

                if (!hasFlag(KeepPlans, TEST_FLAGS)) {
                    fft = Eigen::FFT<SCALAR_T>(Eigen::default_fft_impl<SCALAR_T>(), FFT_FLAGS);
                }
            }
        }
    
    protected:
        DataGenerator<SCALAR_T> real_gen;
        DataGenerator<complex_t> complex_gen;
        
        const std::string get_flags_context() {
            std::stringstream ss;
            ss << std::endl << "Test Flags:" << std::endl;
            ss << "    " << std::left << std::setw(30) << (hasFlag(Preallocate, TEST_FLAGS) ? "Preallocated output matrices" : "Unallocated output matrices") << std::endl;
            ss << "    " << std::left << std::setw(30) << (hasFlag(UnaryFFT, TEST_FLAGS) ? "Unary fwd/inv functions" : "Binary fwd/inv functions") << std::endl;
            ss << "    " << std::left << std::setw(30) << (hasFlag(KeepPlans, TEST_FLAGS) ? "No clearing of plans between runs" : "Clearing plans between runs") << std::endl;
            ss << "    " << std::left << std::setw(30) << (hasFlag(Real, TEST_FLAGS) ? "r2c/c2r" : "c2c") << std::endl;
            ss << "    " << std::left << std::setw(30) << (hasFlag(InPlace, TEST_FLAGS) ? "Inplace" : "Outplace") << std::endl;
            ss << "    " << std::left << std::setw(30) << (hasFlag(UseVectors, TEST_FLAGS) ? "Use Eigen::Vector" : "Use Eigen::Matrix") << std::endl;
            
            ss << "FFT Flags:" << std::endl;
            ss << "    " << std::left << std::setw(30) << (hasFlag(Unscaled, FFT_FLAGS) ? "Unscaled FFT" : "Scaled FFT") << std::endl;
            ss << "    " << std::left << std::setw(30) << (hasFlag(HalfSpectrum, FFT_FLAGS) ? "HalfSpectrum FFT" : "FullSpectrum FFT") << std::endl;
            
            return ss.str();
        }
};

template <typename T>
auto real_const(T, T) {
    return T(1.);
}

template <typename T>
auto complex_const(std::complex<T>) {
    return std::complex<T>(1., 1.);
}

template <typename T>
auto sin2d(T x, T y) {
    return std::sin(x) + std::sin(y);
}

namespace utf = boost::unit_test;

// Main test suite
BOOST_AUTO_TEST_SUITE(EigenFFTTests)

// Macro to create test cases with different flag combinations
#define TEST_FFT_CONFIG_RT_FUNC(scalar_type, fft_flags, test_flags, test_name, sizes_func) \
        FFTTest<scalar_type, fft_flags, Default, test_flags> tester; \
        BOOST_AUTO_TEST_CASE(test_name##_sin, *utf::description("Testing Roundtrip FFT with " #scalar_type " and sin signal")) { \
            if constexpr (hasFlag(Real, test_flags)) { \
                tester.test_roundtrip_func(TestDimensions::sizes_func(), sin2d<scalar_type>); \
            } else { \
                tester.test_roundtrip_func(TestDimensions::sizes_func(), std::sin<std::complex<scalar_type>>); \
            } \
        } \
        BOOST_AUTO_TEST_CASE(test_name##_const, *utf::description("Testing Roundtrip FFT with " #scalar_type " and const signal")) { \
            if constexpr (hasFlag(Real, test_flags)) { \
                tester.test_roundtrip_func(TestDimensions::sizes_func(), real_const<scalar_type>); \
            } else { \
                tester.test_roundtrip_func(TestDimensions::sizes_func(), complex_const<scalar_type>); \
            } \
        }


// Test all combinations of flags for vectors of float precision

// Clean up macro
#undef TEST_FFT_CONFIG_RT_FUNC

BOOST_AUTO_TEST_SUITE_END() // EigenFFTTests