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
// TODO: ADD FLAG FOR nfft ARGUMENT
template <typename SCALAR_T, flag_t FFT_FLAGS=Default, flag_t MATRIX_FLAGS=Default, flag_t TEST_FLAGS=Default> 
class FFTTest {
    public:
        using complex_t = std::complex<SCALAR_T>;

        template <typename FUNC_T>
        void initialise_data(const size_t rows, const size_t cols, FUNC_T f) {
            // input generation
            if constexpr (hasFlag(Real, TEST_FLAGS)) {
                Eigen::MatrixX<std::pair<SCALAR_T, SCALAR_T>> coords = real_gen.generateCoordinateMatrix(rows, 1);
                input_real_v.resize(rows);
                for (size_t i = 0; i < rows; i++) {
                    auto& [x, y] = coords(i, 0);
                    input_real_v(i, 0) = f(x, y);
                }
                
                if constexpr (hasFlag(Unscaled, FFT_FLAGS)) {
                    input_real_v.normalize();
                }

                output_log << "Input real vector [first 5 elements]:" << std::endl;
                for (int i = 0; i < std::min(5, static_cast<int>(rows)); i++) {
                    output_log << "  " << i << ": " << input_real_v(i) << std::endl;
                }
            }
            else {
                input_complex_v = complex_gen.generateCoordinateMatrix(rows, 1);
                input_complex_v.unaryExpr(f);

                if constexpr (hasFlag(Unscaled, FFT_FLAGS)) {
                    input_complex_v.normalize();
                }
                
                output_log << "Input complex vector [first 5 elements]:" << std::endl;
                for (int i = 0; i < std::min(5, static_cast<int>(rows)); i++) {
                    output_log << "  " << i << ": " << input_complex_v(i).real() << " + " << input_complex_v(i).imag() << "i" << std::endl;
                }
            }

            // Frequency vector
            if constexpr (hasFlag(Preallocate, TEST_FLAGS)) {
                if constexpr (hasFlag(HalfSpectrum, FFT_FLAGS)) {
                    freq_v = Eigen::VectorX<complex_t>(rows/2 + 1); // TODO: InPlace also needs padding.
                }
                else {
                    freq_v = Eigen::VectorX<complex_t>(rows);
                }

                if constexpr (hasFlag(Real, TEST_FLAGS)) {
                    output_real_v = Eigen::VectorX<SCALAR_T>(rows);
                }
                else {
                    output_complex_v = Eigen::VectorX<complex_t>(rows);
                }
            }
        }

        template <typename FUNC_T>
        void testRoundTrip(const std::vector<std::pair<size_t, size_t>>& sizes, FUNC_T f) {
            Eigen::FFT<SCALAR_T> fft(Eigen::default_fft_impl<SCALAR_T>(), FFT_FLAGS);

            for (const auto& [rows, cols] : sizes) {
                if constexpr (hasFlag(UseVectors, TEST_FLAGS)) {
                    BOOST_TEST_CONTEXT("Vector size: " << rows << get_flags_context()) {

                        initialise_data(rows, cols, f);

                        if constexpr (hasFlag(InPlace, TEST_FLAGS)) {
                            // TODO
                            output_log << "InPlace is currently not supported yet." << std::endl;
                            BOOST_TEST_MESSAGE(output_log.str());
                            return;
                        }

                        // Actual round trip
                        if constexpr (hasFlag(UnaryFFT, TEST_FLAGS)) {
                            if constexpr (hasFlag(Real, TEST_FLAGS)) {
                                output_log << "c2r using unary operator is currently not supported yet." << std::endl;
                                BOOST_TEST_MESSAGE(output_log.str());
                                return;
                            }
                            else {
                                freq_v = fft.fwd(input_complex_v);
                                output_log << "FFT output [first 5 elements]:" << std::endl;
                                for (int i = 0; i < std::min(5, static_cast<int>(freq_v.size())); i++) {
                                    output_log << "  " << i << ": " << freq_v(i).real() << " + " << freq_v(i).imag() << "i" << std::endl;
                                }
                                
                                output_complex_v = fft.inv(freq_v);
                                if constexpr (hasFlag(Unscaled, FFT_FLAGS)) {
                                    output_complex_v.normalize();
                                }
                                
                                output_log << "IFFT output [first 5 elements]:" << std::endl;
                                for (int i = 0; i < std::min(5, static_cast<int>(output_complex_v.size())); i++) {
                                    output_log << "  " << i << ": " << output_complex_v(i).real() << " + " << output_complex_v(i).imag() << "i" << std::endl;
                                }
                                
                                output_log << "Error magnitude: " << (input_complex_v - output_complex_v).norm() 
                                          << ", Tolerance: " << get_tolerance<SCALAR_T>() << std::endl;
                                
                                BOOST_TEST_MESSAGE(output_log.str());
                                BOOST_CHECK_MESSAGE(input_complex_v.isApprox(output_complex_v, get_tolerance<SCALAR_T>()),
                                                "FFT->IFFT roundtrip failed");
                            }
                        }
                        else {
                            if constexpr (hasFlag(Real, TEST_FLAGS)) {
                                fft.fwd(freq_v, input_real_v);
                                output_log << "FFT output [first 5 elements]:" << std::endl;
                                for (int i = 0; i < std::min(5, static_cast<int>(freq_v.size())); i++) {
                                    output_log << "  " << i << ": " << freq_v(i).real() << " + " << freq_v(i).imag() << "i" << std::endl;
                                }
                                
                                fft.inv(output_real_v, freq_v);
                                if constexpr (hasFlag(Unscaled, FFT_FLAGS)) {
                                    output_real_v.normalize();
                                }

                                output_log << "IFFT output [first 5 elements]:" << std::endl;
                                for (int i = 0; i < std::min(5, static_cast<int>(output_real_v.size())); i++) {
                                    output_log << "  " << i << ": " << output_real_v(i) << std::endl;
                                }
                                
                                output_log << "Error magnitude: " << (input_real_v - output_real_v).norm() 
                                          << ", Tolerance: " << get_tolerance<SCALAR_T>() << std::endl;
                                
                                BOOST_TEST_MESSAGE(output_log.str());
                                BOOST_CHECK_MESSAGE(input_real_v.isApprox(output_real_v, get_tolerance<SCALAR_T>()),
                                                "FFT->IFFT roundtrip failed");
                            }
                            else {
                                fft.fwd(freq_v, input_complex_v);
                                output_log << "FFT output [first 5 elements]:" << std::endl;
                                for (int i = 0; i < std::min(5, static_cast<int>(freq_v.size())); i++) {
                                    output_log << "  " << i << ": " << freq_v(i).real() << " + " << freq_v(i).imag() << "i" << std::endl;
                                }
                                
                                fft.inv(output_complex_v, freq_v);
                                if constexpr (hasFlag(Unscaled, FFT_FLAGS)) {
                                    output_complex_v.normalize();
                                }

                                output_log << "IFFT output [first 5 elements]:" << std::endl;
                                for (int i = 0; i < std::min(5, static_cast<int>(output_complex_v.size())); i++) {
                                    output_log << "  " << i << ": " << output_complex_v(i).real() << " + " << output_complex_v(i).imag() << "i" << std::endl;
                                }
                                
                                output_log << "Error magnitude: " << (input_complex_v - output_complex_v).norm() 
                                          << ", Tolerance: " << get_tolerance<SCALAR_T>() << std::endl;
                                
                                BOOST_TEST_MESSAGE(output_log.str());
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

                clear_data();
            }
        }
        
        template <typename FUNC_T>
        void test_fwd_func(const std::vector<std::pair<size_t, size_t>>& sizes, FUNC_T f) {
            
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
        
        void clear_data() {
            input_real = {};
            input_complex = {};
            output_real = {};
            output_complex = {};
            freq = {};
            input_real_v = {};
            input_complex_v = {};
            output_real_v = {};
            output_complex_v = {};
            freq_v = {};
            output_log.clear();
        }

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
        std::stringstream output_log;
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

// TODO: improve API here, no need for rand to sit in DataGenerator anymore...
// in fact, only one method in DataGenerator is needed now.
template <typename T>
auto random(T, T) {
    return DataGenerator<T>::template rand<T>();
}

template <typename T>
auto random(std::complex<T>) {
    return DataGenerator<std::complex<T>>::template rand<std::complex<T>>();
}

namespace utf = boost::unit_test;

// Main test suite
BOOST_AUTO_TEST_SUITE(EigenFFTTests)

// Macro to create test cases with deterministic test functions
#define TEST_FFT_CONFIG_RT_FUNC(scalar_type, fft_flags, test_flags, test_name, sizes_func) \
        FFTTest<scalar_type, fft_flags, Default, test_flags> tester_##test_name; \
        BOOST_AUTO_TEST_CASE(test_name##_sin, *utf::description("Testing Roundtrip FFT with " #scalar_type " and sin signal")) { \
            if constexpr (hasFlag(Real, test_flags)) { \
                tester_##test_name.testRoundTrip(TestDimensions::sizes_func(), sin2d<scalar_type>); \
            } else { \
                tester_##test_name.testRoundTrip(TestDimensions::sizes_func(), std::sin<std::complex<scalar_type>>); \
            } \
        } \
        BOOST_AUTO_TEST_CASE(test_name##_const, *utf::description("Testing Roundtrip FFT with " #scalar_type " and const signal")) { \
            if constexpr (hasFlag(Real, test_flags)) { \
                tester_##test_name.testRoundTrip(TestDimensions::sizes_func(), real_const<scalar_type>); \
            } else { \
                tester_##test_name.testRoundTrip(TestDimensions::sizes_func(), complex_const<scalar_type>); \
            } \
        }

// Macro for random test functions that run multiple iterations
#define TEST_FFT_CONFIG_RT_FUNC_RANDOM(scalar_type, fft_flags, test_flags, test_name, sizes_func, iterations) \
        FFTTest<scalar_type, fft_flags, Default, test_flags> random_tester_##test_name; \
        BOOST_AUTO_TEST_CASE(test_name##_random, *utf::description("Testing Roundtrip FFT with " #scalar_type " and random signal")) { \
            for (int iter = 0; iter < iterations; ++iter) { \
                BOOST_TEST_CONTEXT("Random iteration " << iter + 1 << " of " << iterations) { \
                    if constexpr (hasFlag(Real, test_flags)) { \
                        random_tester_##test_name.testRoundTrip(TestDimensions::sizes_func(), \
                            [](scalar_type x, scalar_type y) { return random(x, y); }); \
                    } else { \
                        random_tester_##test_name.testRoundTrip(TestDimensions::sizes_func(), \
                            [](std::complex<scalar_type> z) { return random(z); }); \
                    } \
                } \
            } \
        }

// Group 1: Float Complex-to-Complex Vector Tests
BOOST_AUTO_TEST_SUITE(FloatComplexToComplexVectorTests)

// Basic tests
TEST_FFT_CONFIG_RT_FUNC(float, Default, Default | UseVectors, float_c2c_basic, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(float, Default, Default | UseVectors, float_c2c_basic, getSimpleSizes, 5)

// Unary FFT function tests
TEST_FFT_CONFIG_RT_FUNC(float, Default, UnaryFFT | UseVectors, float_c2c_unary, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(float, Default, UnaryFFT | UseVectors, float_c2c_unary, getSimpleSizes, 3)

// Preallocate tests
TEST_FFT_CONFIG_RT_FUNC(float, Default, Preallocate | UseVectors, float_c2c_prealloc, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(float, Default, Preallocate | UseVectors, float_c2c_prealloc, getSimpleSizes, 3)

// Keep plans tests
TEST_FFT_CONFIG_RT_FUNC(float, Default, KeepPlans | UseVectors, float_c2c_keepplans, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(float, Default, KeepPlans | UseVectors, float_c2c_keepplans, getSimpleSizes, 3)

BOOST_AUTO_TEST_SUITE_END()

// Group 2: Float Complex-to-Complex with FFT flags
BOOST_AUTO_TEST_SUITE(FloatComplexToComplexVectorFFTFlagsTests)

// Unscaled FFT tests
TEST_FFT_CONFIG_RT_FUNC(float, Unscaled, Default | UseVectors, float_c2c_unscaled, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(float, Unscaled, Default | UseVectors, float_c2c_unscaled, getSimpleSizes, 3)

// Half-spectrum tests
TEST_FFT_CONFIG_RT_FUNC(float, HalfSpectrum, Default | UseVectors, float_c2c_halfspectrum, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(float, HalfSpectrum, Default | UseVectors, float_c2c_halfspectrum, getSimpleSizes, 3)

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(FloatComplexToComplexVectorFFTMixedFlagsTests)

// Combined flags tests
TEST_FFT_CONFIG_RT_FUNC(float, Unscaled | HalfSpectrum, Default | UseVectors, float_c2c_unscaled_halfspectrum, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(float, Unscaled | HalfSpectrum, Default | UseVectors, float_c2c_unscaled_halfspectrum, getSimpleSizes, 3)

BOOST_AUTO_TEST_SUITE_END()

// Group 3: Float Real-to-Complex Vector Tests
BOOST_AUTO_TEST_SUITE(FloatRealToComplexVectorTests)

// Basic real tests
TEST_FFT_CONFIG_RT_FUNC(float, Default, Real | UseVectors, float_r2c_basic, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(float, Default, Real | UseVectors, float_r2c_basic, getSimpleSizes, 5)

// Unary FFT real tests
TEST_FFT_CONFIG_RT_FUNC(float, Default, Real | UnaryFFT | UseVectors, float_r2c_unary, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(float, Default, Real | UnaryFFT | UseVectors, float_r2c_unary, getSimpleSizes, 3)

// Preallocate real tests
TEST_FFT_CONFIG_RT_FUNC(float, Default, Real | Preallocate | UseVectors, float_r2c_prealloc, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(float, Default, Real | Preallocate | UseVectors, float_r2c_prealloc, getSimpleSizes, 3)

// Keep plans real tests
TEST_FFT_CONFIG_RT_FUNC(float, Default, Real | KeepPlans | UseVectors, float_r2c_keepplans, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(float, Default, Real | KeepPlans | UseVectors, float_r2c_keepplans, getSimpleSizes, 3)

BOOST_AUTO_TEST_SUITE_END()

// Group 4: Float Real-to-Complex with FFT flags
BOOST_AUTO_TEST_SUITE(FloatRealToComplexVectorFFTFlagsTests)

// Unscaled real tests
TEST_FFT_CONFIG_RT_FUNC(float, Unscaled, Real | UseVectors, float_r2c_unscaled, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(float, Unscaled, Real | UseVectors, float_r2c_unscaled, getSimpleSizes, 3)

// Half-spectrum real tests
TEST_FFT_CONFIG_RT_FUNC(float, HalfSpectrum, Real | UseVectors, float_r2c_halfspectrum, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(float, HalfSpectrum, Real | UseVectors, float_r2c_halfspectrum, getSimpleSizes, 3)

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(FloatRealToComplexVectorFFTMixedFlagsTests)

// Combined flags real tests
TEST_FFT_CONFIG_RT_FUNC(float, Unscaled | HalfSpectrum, Real | UseVectors, float_r2c_unscaled_halfspectrum, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(float, Unscaled | HalfSpectrum, Real | UseVectors, float_r2c_unscaled_halfspectrum, getSimpleSizes, 3)

BOOST_AUTO_TEST_SUITE_END()

// Group 5: Double Complex-to-Complex Vector Tests
BOOST_AUTO_TEST_SUITE(DoubleComplexToComplexVectorTests)

// Basic tests
TEST_FFT_CONFIG_RT_FUNC(double, Default, Default | UseVectors, double_c2c_basic, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(double, Default, Default | UseVectors, double_c2c_basic, getSimpleSizes, 5)

// Unary FFT function tests
TEST_FFT_CONFIG_RT_FUNC(double, Default, UnaryFFT | UseVectors, double_c2c_unary, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(double, Default, UnaryFFT | UseVectors, double_c2c_unary, getSimpleSizes, 3)

// Preallocate tests
TEST_FFT_CONFIG_RT_FUNC(double, Default, Preallocate | UseVectors, double_c2c_prealloc, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(double, Default, Preallocate | UseVectors, double_c2c_prealloc, getSimpleSizes, 3)

// Keep plans tests
TEST_FFT_CONFIG_RT_FUNC(double, Default, KeepPlans | UseVectors, double_c2c_keepplans, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(double, Default, KeepPlans | UseVectors, double_c2c_keepplans, getSimpleSizes, 3)

BOOST_AUTO_TEST_SUITE_END()

// Group 6: Double Complex-to-Complex with FFT flags
BOOST_AUTO_TEST_SUITE(DoubleComplexToComplexFFTFlagsTests)

// Unscaled FFT tests
TEST_FFT_CONFIG_RT_FUNC(double, Unscaled, Default | UseVectors, double_c2c_unscaled, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(double, Unscaled, Default | UseVectors, double_c2c_unscaled, getSimpleSizes, 3)

// Half-spectrum tests
TEST_FFT_CONFIG_RT_FUNC(double, HalfSpectrum, Default | UseVectors, double_c2c_halfspectrum, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(double, HalfSpectrum, Default | UseVectors, double_c2c_halfspectrum, getSimpleSizes, 3)

// Combined flags tests
TEST_FFT_CONFIG_RT_FUNC(double, Unscaled | HalfSpectrum, Default | UseVectors, double_c2c_unscaled_halfspectrum, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(double, Unscaled | HalfSpectrum, Default | UseVectors, double_c2c_unscaled_halfspectrum, getSimpleSizes, 3)

BOOST_AUTO_TEST_SUITE_END()

// Group 7: Double Real-to-Complex Vector Tests
BOOST_AUTO_TEST_SUITE(DoubleRealToComplexVectorTests)

// Basic real tests
TEST_FFT_CONFIG_RT_FUNC(double, Default, Real | UseVectors, double_r2c_basic, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(double, Default, Real | UseVectors, double_r2c_basic, getSimpleSizes, 5)

// Unary FFT real tests
TEST_FFT_CONFIG_RT_FUNC(double, Default, Real | UnaryFFT | UseVectors, double_r2c_unary, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(double, Default, Real | UnaryFFT | UseVectors, double_r2c_unary, getSimpleSizes, 3)

// Preallocate real tests
TEST_FFT_CONFIG_RT_FUNC(double, Default, Real | Preallocate | UseVectors, double_r2c_prealloc, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(double, Default, Real | Preallocate | UseVectors, double_r2c_prealloc, getSimpleSizes, 3)

// Keep plans real tests
TEST_FFT_CONFIG_RT_FUNC(double, Default, Real | KeepPlans | UseVectors, double_r2c_keepplans, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(double, Default, Real | KeepPlans | UseVectors, double_r2c_keepplans, getSimpleSizes, 3)

BOOST_AUTO_TEST_SUITE_END()

// Group 8: Double Real-to-Complex with FFT flags
BOOST_AUTO_TEST_SUITE(DoubleRealToComplexVectorFFTFlagsTests)

// Unscaled real tests
TEST_FFT_CONFIG_RT_FUNC(double, Unscaled, Real | UseVectors, double_r2c_unscaled, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(double, Unscaled, Real | UseVectors, double_r2c_unscaled, getSimpleSizes, 3)

// Half-spectrum real tests
TEST_FFT_CONFIG_RT_FUNC(double, HalfSpectrum, Real | UseVectors, double_r2c_halfspectrum, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(double, HalfSpectrum, Real | UseVectors, double_r2c_halfspectrum, getSimpleSizes, 3)

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(DoubleRealToComplexVectorFFTMixedFlagsTests)

// Combined flags real tests
TEST_FFT_CONFIG_RT_FUNC(double, Unscaled | HalfSpectrum, Real | UseVectors, double_r2c_unscaled_halfspectrum, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(double, Unscaled | HalfSpectrum, Real | UseVectors, double_r2c_unscaled_halfspectrum, getSimpleSizes, 3)

BOOST_AUTO_TEST_SUITE_END()

// Group 9: Comprehensive Tests with Full Vector Sizes
BOOST_AUTO_TEST_SUITE(ComprehensiveVectorTests)

// Float comprehensive tests
TEST_FFT_CONFIG_RT_FUNC(float, Default, Default | UseVectors, float_c2c_comprehensive, getTestSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(float, Default, Default | UseVectors, float_c2c_comprehensive, getTestSizes, 3)

TEST_FFT_CONFIG_RT_FUNC(float, Default, Real | UseVectors, float_r2c_comprehensive, getTestSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(float, Default, Real | UseVectors, float_r2c_comprehensive, getTestSizes, 3)

// Double comprehensive tests
TEST_FFT_CONFIG_RT_FUNC(double, Default, Default | UseVectors, double_c2c_comprehensive, getTestSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(double, Default, Default | UseVectors, double_c2c_comprehensive, getTestSizes, 3)

TEST_FFT_CONFIG_RT_FUNC(double, Default, Real | UseVectors, double_r2c_comprehensive, getTestSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(double, Default, Real | UseVectors, double_r2c_comprehensive, getTestSizes, 3)

BOOST_AUTO_TEST_SUITE_END()

// Clean up macros
#undef TEST_FFT_CONFIG_RT_FUNC
#undef TEST_FFT_CONFIG_RT_FUNC_RANDOM

BOOST_AUTO_TEST_SUITE_END() // EigenFFTTests