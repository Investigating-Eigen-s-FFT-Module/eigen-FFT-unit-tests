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

#define RAND_RUNS 100

using flag_t = int;
static const flag_t Default = 0;
static const flag_t Unscaled = 1;
static const flag_t HalfSpectrum = 2;
static const flag_t Speedy = 32767;

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

constexpr inline bool hasFlag(flag_t flag, flag_t flags) {
    return static_cast<bool>(flags & flag);
}

// TODO: IMPLEMENT MATRIX TEMPLATE FLAGS
// TODO: ADD FLAG FOR nfft ARGUMENT
template <typename SCALAR_T, flag_t FFT_FLAGS=Default, flag_t MATRIX_FLAGS=Default, flag_t TEST_FLAGS=Default> 
class FFTTest {
    public:
        using complex_t = std::complex<SCALAR_T>;
        using big_float_t = long double;
        using big_complex_t = std::complex<big_float_t>;

        template <typename FUNC_T>
        void initialiseData(const size_t rows, const size_t cols, FUNC_T f) {
            // VECTOR INPUT INITIALISATION
            if constexpr (hasFlag(UseVectors, TEST_FLAGS)) {
                // REAL VECTORS
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
                // COMPLEX VECTORS
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

                // FREQUENCY VECTOR
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
            // MATRIX INPUT INITIALISATION
            else {
                // REAL MATRICES
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
                // COMPLEX MATRICES
                else {
                    input_complex = complex_gen.generateCoordinateMatrix(rows, cols);
                    input_complex.unaryExpr(f);
                }

                // FREQUENCY MATRIX
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
            }
        }

        template <typename FUNC_T>
        void testRoundTrip(const std::vector<std::pair<size_t, size_t>>& sizes, FUNC_T f) {
            // ONE-TIME INITALISE Eigen::FFT
            fft = Eigen::FFT<SCALAR_T>(Eigen::default_fft_impl<SCALAR_T>(), FFT_FLAGS);
            for (const auto& [rows, cols] : sizes) {
                initialiseData(rows, cols, f);

                // VECTOR TESTING
                if constexpr (hasFlag(UseVectors, TEST_FLAGS)) {
                    BOOST_TEST_CONTEXT("Vector size: " << rows << get_flags_context()) {

                        // INPLACE ROUNDTRIP
                        if constexpr (hasFlag(InPlace, TEST_FLAGS)) {
                            // TODO
                            output_log << "InPlace is currently not supported yet." << std::endl;
                            BOOST_TEST_MESSAGE(output_log.str());
                            return;
                        }

                        // UNARY ROUNDTRIP
                        if constexpr (hasFlag(UnaryFFT, TEST_FLAGS)) {
                            // REAL UNARY ROUNDTRIP
                            if constexpr (hasFlag(Real, TEST_FLAGS)) {
                                // TODO
                                output_log << "c2r using unary operator is currently not supported yet." << std::endl;
                                BOOST_TEST_MESSAGE(output_log.str());
                                return;
                            }
                            // COMPLEX UNARY ROUNDTRIP
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
                        // BINARY ROUNDTRIP
                        else {
                            // REAL BINARY ROUNDTRIP
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
                            // COMPLEX BINARY ROUNDTRIP
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
                // VECTOR TESTING
                else {
                    BOOST_TEST_CONTEXT("Matrix size: " << rows << "x" << cols << get_flags_context()) {

                        // INPLACE ROUNDTRIP
                        if constexpr (hasFlag(InPlace, TEST_FLAGS)) {
                            // TODO
                            BOOST_TEST_MESSAGE("InPlace is currently not supported yet.");
                            return;
                        }
    
                        // UNARY ROUNDTRIP
                        if constexpr (hasFlag(UnaryFFT, TEST_FLAGS)) {
                            // REAL UNARY ROUNDTRIP
                            if constexpr (hasFlag(Real, TEST_FLAGS)) {
                                // TODO
                                BOOST_TEST_MESSAGE("c2r using unary operator is currently not supported yet.");
                                return;
                                // freq = fft.fwd(input_real);
                                // output_real = fft.inv(freq);
                                // BOOST_CHECK_MESSAGE(input_real.isApprox(output_real, get_tolerance<SCALAR_T>()),
                                //                     "FFT->IFFT roundtrip failed");
                            }
                            // COMPLEX UNARY ROUNDTRIP
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
                        // BINARY ROUNDTRIP
                        else {
                            // REAL BINARY ROUNDTRIP
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
                            // COMPLEX BINARY ROUNDTRIP
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

                // CLEAR PLANS
                if (!hasFlag(KeepPlans, TEST_FLAGS)) {
                    fft = Eigen::FFT<SCALAR_T>(Eigen::default_fft_impl<SCALAR_T>(), FFT_FLAGS);
                }

                clear_data();
            } // for loop
        }
        
        template <typename FUNC_T>
        void testFFT(const std::vector<std::pair<size_t, size_t>>& sizes, FUNC_T f) {
            // ONE-TIME INITALISE Eigen::FFT
            fft = Eigen::FFT<SCALAR_T>(Eigen::default_fft_impl<SCALAR_T>(), FFT_FLAGS);
            for (const auto& [rows, cols] : sizes) {
                initialiseData(rows, cols, f);
                // VECTOR TESTING
                if constexpr (hasFlag(UseVectors, TEST_FLAGS)) {
                    BOOST_TEST_CONTEXT("Vector size: " << rows << get_flags_context()) {
                        // INPLACE FFT
                        if constexpr (hasFlag(InPlace, TEST_FLAGS)) {
                            // TODO
                            output_log << "InPlace is currently not supported yet." << std::endl;
                            BOOST_TEST_MESSAGE(output_log.str());
                            return;
                        }

                        // UNARY FFT
                        if constexpr (hasFlag(UnaryFFT, TEST_FLAGS)) {\
                            // UNARY REAL FFT
                            if constexpr (hasFlag(Real, TEST_FLAGS)) {
                                // TODO
                                output_log << "c2r using unary operator is currently not supported yet." << std::endl;
                                BOOST_TEST_MESSAGE(output_log.str());
                                return;
                            }
                            // UNARY COMPLEX FFT
                            else {
                                freq_v = fft.fwd(input_complex_v);
                                output_log << "FFT output [first 5 elements]:" << std::endl;
                                for (int i = 0; i < std::min(5, static_cast<int>(freq_v.size())); i++) {
                                    output_log << "  " << i << ": " << freq_v(i).real() << " + " << freq_v(i).imag() << "i" << std::endl;
                                }
                                
                                freq_oracle_v = oracle_fwd(input_complex_v);
                                output_log << "FFT Oracle output [first 5 elements]:" << std::endl;
                                for (int i = 0; i < std::min(5, static_cast<int>(freq_oracle_v.size())); i++) {
                                    output_log << "  " << i << ": " << freq_oracle_v(i).real() << " + " << freq_oracle_v(i).imag() << "i" << std::endl;
                                }
                                
                                output_log << "Error magnitude: " << (freq_v - freq_oracle_v).norm() 
                                          << ", Tolerance: " << get_tolerance<SCALAR_T>() << std::endl;
                                
                                BOOST_TEST_MESSAGE(output_log.str());
                                BOOST_CHECK_MESSAGE(freq_v.isApprox(freq_oracle_v, get_tolerance<SCALAR_T>()),
                                                "FFT failed");
                            }
                        }
                        // BINARY FFT
                        else {
                            // BINARY REAL FFT
                            if constexpr (hasFlag(Real, TEST_FLAGS)) {
                                fft.fwd(freq_v, input_real_v);
                                output_log << "FFT output [first 5 elements]:" << std::endl;
                                for (int i = 0; i < std::min(5, static_cast<int>(freq_v.size())); i++) {
                                    output_log << "  " << i << ": " << freq_v(i).real() << " + " << freq_v(i).imag() << "i" << std::endl;
                                }
                                
                                freq_oracle_v = oracle_fwd(input_real_v);
                                output_log << "FFT Oracle output [first 5 elements]:" << std::endl;
                                for (int i = 0; i < std::min(5, static_cast<int>(freq_oracle_v.size())); i++) {
                                    output_log << "  " << i << ": " << freq_oracle_v(i).real() << " + " << freq_oracle_v(i).imag() << "i" << std::endl;
                                }

                                output_log << "Error magnitude: " << (freq_v - freq_oracle_v).norm() 
                                          << ", Tolerance: " << get_tolerance<SCALAR_T>() << std::endl;
                                
                                BOOST_TEST_MESSAGE(output_log.str());
                                BOOST_CHECK_MESSAGE(freq_v.isApprox(freq_oracle_v, get_tolerance<SCALAR_T>()),
                                                "FFT failed");
                            }
                            // BINARY COMPLEX FFT
                            else {
                                fft.fwd(freq_v, input_complex_v);
                                output_log << "FFT output [first 5 elements]:" << std::endl;
                                for (int i = 0; i < std::min(5, static_cast<int>(freq_v.size())); i++) {
                                    output_log << "  " << i << ": " << freq_v(i).real() << " + " << freq_v(i).imag() << "i" << std::endl;
                                }
                                
                                freq_oracle_v = oracle_fwd(input_complex_v);
                                output_log << "FFT Oracle output [first 5 elements]:" << std::endl;
                                for (int i = 0; i < std::min(5, static_cast<int>(freq_oracle_v.size())); i++) {
                                    output_log << "  " << i << ": " << freq_oracle_v(i).real() << " + " << freq_oracle_v(i).imag() << "i" << std::endl;
                                }

                                output_log << "Error magnitude: " << (freq_v - freq_oracle_v).norm() 
                                          << ", Tolerance: " << get_tolerance<SCALAR_T>() << std::endl;
                                
                                BOOST_TEST_MESSAGE(output_log.str());
                                BOOST_CHECK_MESSAGE(freq_v.isApprox(freq_oracle_v, get_tolerance<SCALAR_T>()),
                                                "FFT failed");
                            }
                        }
                    }
                }
                // MATRIX TESTING
                else {
                    BOOST_TEST_CONTEXT("Matrix size: " << rows << "x" << cols << get_flags_context()) {
                        if constexpr (hasFlag(InPlace, TEST_FLAGS)) {
                            // TODO
                            BOOST_TEST_MESSAGE("InPlace is currently not supported yet.");
                            return;
                        }
    
                        // Actual fwd
                        if constexpr (hasFlag(UnaryFFT, TEST_FLAGS)) {
                            if constexpr (hasFlag(Real, TEST_FLAGS)) {
                                // TODO
                                BOOST_TEST_MESSAGE("c2r using unary operator is currently not supported yet.");
                                return;
                            }
                            else {
                                // TODO
                                BOOST_TEST_MESSAGE("Matrix input is currently not supported yet.");
                                return;
                                // freq = fft.fwd(input_complex);
                            }
                        }
                        else {
                            if constexpr (hasFlag(Real, TEST_FLAGS)) {
                                BOOST_TEST_MESSAGE("Matrix input is currently not supported yet.");
                                return;
                                // TODO: allow once matrix input is fine
                            }
                            else {
                                BOOST_TEST_MESSAGE("Matrix input is currently not supported yet.");
                                return;
                                // TODO: allow once matrix input is fine
                            }
                        }
                    }
                }

                // CLEAR PLANS
                if (!hasFlag(KeepPlans, TEST_FLAGS)) {
                    fft = Eigen::FFT<SCALAR_T>(Eigen::default_fft_impl<SCALAR_T>(), FFT_FLAGS);
                }

                clear_data();
            }
        }

        template <typename FUNC_T>
        void testIFFT(const std::vector<std::pair<size_t, size_t>>& sizes, FUNC_T f) {
            // ONE-TIME INITIALISE Eigen::FFT
            fft = Eigen::FFT<SCALAR_T>(Eigen::default_fft_impl<SCALAR_T>(), FFT_FLAGS);
            for (const auto& [rows, cols] : sizes) {
                initialiseData(rows, cols, f);

                // VECTOR TESTING
                if constexpr (hasFlag(UseVectors, TEST_FLAGS)) {
                    BOOST_TEST_CONTEXT("Vector size: " << rows << get_flags_context()) {
                        // INPLACE IFFT
                        if constexpr (hasFlag(InPlace, TEST_FLAGS)) {
                            // TODO
                            output_log << "InPlace is currently not supported yet." << std::endl;
                            BOOST_TEST_MESSAGE(output_log.str());
                            return;
                        }

                        // UNARY IFFT
                        if constexpr (hasFlag(UnaryFFT, TEST_FLAGS)) {
                            // UNARY REAL IFFT
                            if constexpr (hasFlag(Real, TEST_FLAGS)) {
                                // TODO
                                output_log << "r2c using unary operator is currently not supported yet." << std::endl;
                                BOOST_TEST_MESSAGE(output_log.str());
                                return;
                            }
                            // UNARY COMPLEX FFT
                            else {
                                // bit hacky but I don't want to redesign data generation, so do an oracle fft
                                freq_v = oracle_fwd(input_complex_v);
                                output_log << "Preliminary FFT output [first 5 elements]:" << std::endl;
                                for (int i = 0; i < std::min(5, static_cast<int>(freq_v.size())); i++) {
                                    output_log << "  " << i << ": " << freq_v(i).real() << " + " << freq_v(i).imag() << "i" << std::endl;
                                }

                                output_complex_v = fft.inv(freq_v);
                                output_log << "IFFT output [first 5 elements]:" << std::endl;
                                for (int i = 0; i < std::min(5, static_cast<int>(output_complex_v.size())); i++) {
                                    output_log << "  " << i << ": " << output_complex_v(i).real() << " + " << output_complex_v(i).imag() << "i" << std::endl;
                                }
                                
                                output_complex_oracle_v = oracle_inv(freq_v);
                                output_log << "IFFT Oracle output [first 5 elements]:" << std::endl;
                                for (int i = 0; i < std::min(5, static_cast<int>(output_complex_oracle_v.size())); i++) {
                                    output_log << "  " << i << ": " << output_complex_oracle_v(i).real() << " + " << output_complex_oracle_v(i).imag() << "i" << std::endl;
                                }
                                
                                output_log << "Error magnitude: " << (output_complex_v - output_complex_oracle_v).norm() 
                                          << ", Tolerance: " << get_tolerance<SCALAR_T>() << std::endl;
                                
                                BOOST_TEST_MESSAGE(output_log.str());
                                BOOST_CHECK_MESSAGE(output_complex_v.isApprox(output_complex_oracle_v, get_tolerance<SCALAR_T>()),
                                                "IFFT failed");
                            }
                        }
                        // BINARY IFFT
                        else {
                            // BINARY REAL IFFT
                            if constexpr (hasFlag(Real, TEST_FLAGS)) {
                                // bit hacky but I don't want to redesign data generation, so do an oracle fft
                                freq_v = oracle_fwd(input_real_v);
                                output_log << "Preliminary FFT output [first 5 elements]:" << std::endl;
                                for (int i = 0; i < std::min(5, static_cast<int>(freq_v.size())); i++) {
                                    output_log << "  " << i << ": " << freq_v(i).real() << " + " << freq_v(i).imag() << "i" << std::endl;
                                }

                                fft.inv(output_real_v, freq_v);
                                output_log << "IFFT output [first 5 elements]:" << std::endl;
                                for (int i = 0; i < std::min(5, static_cast<int>(output_real_v.size())); i++) {
                                    output_log << "  " << i << ": " << output_real_v(i) << std::endl;
                                }
                                
                                output_real_oracle_v = oracle_inv(freq_v);
                                output_log << "IFFT Oracle output [first 5 elements]:" << std::endl;
                                for (int i = 0; i < std::min(5, static_cast<int>(output_real_oracle_v.size())); i++) {
                                    output_log << "  " << i << ": " << output_real_oracle_v(i) << std::endl;
                                }
                                
                                output_log << "Error magnitude: " << (output_real_v - output_real_oracle_v).norm() 
                                          << ", Tolerance: " << get_tolerance<SCALAR_T>() << std::endl;
                                
                                BOOST_TEST_MESSAGE(output_log.str());
                                BOOST_CHECK_MESSAGE(output_real_v.isApprox(output_real_oracle_v, get_tolerance<SCALAR_T>()),
                                                "IFFT failed");
                            }
                            // BINARY COMPLEX IFFT
                            else {
                                // bit hacky but I don't want to redesign data generation, so do an oracle fft
                                freq_v = oracle_fwd(input_complex_v);
                                output_log << "Preliminary FFT output [first 5 elements]:" << std::endl;
                                for (int i = 0; i < std::min(5, static_cast<int>(freq_v.size())); i++) {
                                    output_log << "  " << i << ": " << freq_v(i).real() << " + " << freq_v(i).imag() << "i" << std::endl;
                                }

                                fft.inv(output_complex_v, freq_v);
                                output_log << "IFFT output [first 5 elements]:" << std::endl;
                                for (int i = 0; i < std::min(5, static_cast<int>(output_complex_v.size())); i++) {
                                    output_log << "  " << i << ": " << output_complex_v(i).real() << " + " << output_complex_v(i).imag() << "i" << std::endl;
                                }
                                
                                output_complex_oracle_v = oracle_inv(freq_v);
                                output_log << "IFFT Oracle output [first 5 elements]:" << std::endl;
                                for (int i = 0; i < std::min(5, static_cast<int>(output_complex_oracle_v.size())); i++) {
                                    output_log << "  " << i << ": " << output_complex_oracle_v(i).real() << " + " << output_complex_oracle_v(i).imag() << "i" << std::endl;
                                }
                                
                                output_log << "Error magnitude: " << (output_complex_v - output_complex_oracle_v).norm() 
                                          << ", Tolerance: " << get_tolerance<SCALAR_T>() << std::endl;
                                
                                BOOST_TEST_MESSAGE(output_log.str());
                                BOOST_CHECK_MESSAGE(output_complex_v.isApprox(output_complex_oracle_v, get_tolerance<SCALAR_T>()),
                                                "IFFT failed");
                            }
                        }
                    }
                }
                // MATRIX TESTING
                else {
                    BOOST_TEST_CONTEXT("Matrix size: " << rows << "x" << cols << get_flags_context()) {
                        if constexpr (hasFlag(InPlace, TEST_FLAGS)) {
                            // TODO
                            BOOST_TEST_MESSAGE("InPlace is currently not supported yet.");
                            return;
                        }
    
                        // Actual fwd
                        if constexpr (hasFlag(UnaryFFT, TEST_FLAGS)) {
                            if constexpr (hasFlag(Real, TEST_FLAGS)) {
                                // TODO
                                BOOST_TEST_MESSAGE("c2r using unary operator is currently not supported yet.");
                                return;
                            }
                            else {
                                // TODO
                                BOOST_TEST_MESSAGE("Matrix input is currently not supported yet.");
                                return;
                                // freq = fft.fwd(input_complex);
                            }
                        }
                        else {
                            if constexpr (hasFlag(Real, TEST_FLAGS)) {
                                BOOST_TEST_MESSAGE("Matrix input is currently not supported yet.");
                                return;
                                // TODO: allow once matrix input is fine
                            }
                            else {
                                BOOST_TEST_MESSAGE("Matrix input is currently not supported yet.");
                                return;
                                // TODO: allow once matrix input is fine
                            }
                        }
                    }
                }
                // CLEAR PLANS
                if (!hasFlag(KeepPlans, TEST_FLAGS)) {
                    fft = Eigen::FFT<SCALAR_T>(Eigen::default_fft_impl<SCALAR_T>(), FFT_FLAGS);
                }

                clear_data();
            }
        }

    protected:
    
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
            output_real_oracle = {};
            output_complex_oracle = {};
            freq = {};
            freq_oracle = {};
            input_real_v = {};
            input_complex_v = {};
            output_real_v = {};
            output_complex_v = {};
            freq_v = {};
            output_real_oracle_v = {};
            output_complex_oracle_v = {};
            freq_oracle_v = {};
            output_log.clear();
        }
        
        // src can be Eigen::MatrixX<SCALAR_T>; implicit conversion works for same scalar types
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
        
            // no scaling on fwd atm, also why it's not sqrt factor.
            // if (!hasFlag(Unscaled, flags)) {
            //     dst /= nfft0 * nfft1;
            // }
        
            Eigen::MatrixX<std::complex<SCALAR_T>> dst_cp = dst.cast<std::complex<SCALAR_T>>();
        
            return dst_cp;
        }
        
        auto oracle_inv(const Eigen::MatrixX<std::complex<SCALAR_T>>& src) {
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
        
            if (!hasFlag(Unscaled, FFT_FLAGS)) {
                dst /= nfft0 * nfft1;
            }
        
            if constexpr (hasFlag(Real, TEST_FLAGS)) {
                Eigen::MatrixX<SCALAR_T> dst_cp = dst.real().template cast<SCALAR_T>();
                return dst_cp;
            }
            else {
                Eigen::MatrixX<std::complex<SCALAR_T>> dst_cp = dst.template cast<std::complex<SCALAR_T>>();
                return dst_cp;
            }
        }
        
        DataGenerator<SCALAR_T> real_gen;
        DataGenerator<complex_t> complex_gen;
        Eigen::MatrixX<SCALAR_T> input_real;
        Eigen::MatrixX<complex_t> input_complex;
        Eigen::MatrixX<SCALAR_T> output_real;
        Eigen::MatrixX<complex_t> output_complex;
        Eigen::MatrixX<SCALAR_T> output_real_oracle;
        Eigen::MatrixX<complex_t> output_complex_oracle;
        Eigen::MatrixX<complex_t> freq;
        Eigen::MatrixX<complex_t> freq_oracle;
        Eigen::VectorX<SCALAR_T> input_real_v;
        Eigen::VectorX<complex_t> input_complex_v;
        Eigen::VectorX<SCALAR_T> output_real_v;
        Eigen::VectorX<complex_t> output_complex_v;
        Eigen::VectorX<complex_t> freq_v;
        Eigen::VectorX<SCALAR_T> output_real_oracle_v;
        Eigen::VectorX<complex_t> output_complex_oracle_v;
        Eigen::VectorX<complex_t> freq_oracle_v;
        Eigen::FFT<SCALAR_T> fft;
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

// Macro to create test cases with deterministic roundtrip test functions
#define TEST_FFT_CONFIG_RT_FUNC(scalar_type, fft_flags, test_flags, test_name, sizes_func) \
        FFTTest<scalar_type, fft_flags, Default, test_flags> tester_##test_name##_rt; \
        BOOST_AUTO_TEST_CASE(test_name##_rt_sin, \
            *utf::label("roundtrip") \
            *utf::label(#scalar_type) \
            *utf::label(hasFlag(Real, test_flags) ? "r2c" : "c2c") \
            *utf::description("Testing Roundtrip FFT with " #scalar_type " and sin signal")) { \
            if constexpr (hasFlag(Real, test_flags)) { \
                tester_##test_name##_rt.testRoundTrip(TestDimensions::sizes_func(), sin2d<scalar_type>); \
            } else { \
                tester_##test_name##_rt.testRoundTrip(TestDimensions::sizes_func(), std::sin<std::complex<scalar_type>>); \
            } \
        } \
        BOOST_AUTO_TEST_CASE(test_name##_rt_const, \
            *utf::label("roundtrip") \
            *utf::label(#scalar_type) \
            *utf::label(hasFlag(Real, test_flags) ? "r2c" : "c2c") \
            *utf::description("Testing Roundtrip FFT with " #scalar_type " and const signal")) { \
            if constexpr (hasFlag(Real, test_flags)) { \
                tester_##test_name##_rt.testRoundTrip(TestDimensions::sizes_func(), real_const<scalar_type>); \
            } else { \
                tester_##test_name##_rt.testRoundTrip(TestDimensions::sizes_func(), complex_const<scalar_type>); \
            } \
        }

// Macro for random roundtrip test functions that run multiple iterations
#define TEST_FFT_CONFIG_RT_FUNC_RANDOM(scalar_type, fft_flags, test_flags, test_name, sizes_func, iterations) \
        FFTTest<scalar_type, fft_flags, Default, test_flags> random_tester_##test_name##_rt; \
        BOOST_AUTO_TEST_CASE(test_name##_rt_random, \
            *utf::label("roundtrip") \
            *utf::label("random") \
            *utf::label(#scalar_type) \
            *utf::label(hasFlag(Real, test_flags) ? "r2c" : "c2c") \
            *utf::description("Testing Roundtrip FFT with " #scalar_type " and random signal")) { \
            for (int iter = 0; iter < iterations; ++iter) { \
                BOOST_TEST_CONTEXT("Random iteration " << iter + 1 << " of " << iterations) { \
                    if constexpr (hasFlag(Real, test_flags)) { \
                        random_tester_##test_name##_rt.testRoundTrip(TestDimensions::sizes_func(), \
                            [](scalar_type x, scalar_type y) { return random(x, y); }); \
                    } else { \
                        random_tester_##test_name##_rt.testRoundTrip(TestDimensions::sizes_func(), \
                            [](std::complex<scalar_type> z) { return random(z); }); \
                    } \
                } \
            } \
        }

// Macro to create test cases with deterministic FFT test functions
#define TEST_FFT_CONFIG_FFT_FUNC(scalar_type, fft_flags, test_flags, test_name, sizes_func) \
        FFTTest<scalar_type, fft_flags, Default, test_flags> tester_##test_name##_fft; \
        BOOST_AUTO_TEST_CASE(test_name##_fft_sin, \
            *utf::label("fft") \
            *utf::label(#scalar_type) \
            *utf::label(hasFlag(Real, test_flags) ? "r2c" : "c2c") \
            *utf::description("Testing FFT with " #scalar_type " and sin signal")) { \
            if constexpr (hasFlag(Real, test_flags)) { \
                tester_##test_name##_fft.testFFT(TestDimensions::sizes_func(), sin2d<scalar_type>); \
            } else { \
                tester_##test_name##_fft.testFFT(TestDimensions::sizes_func(), std::sin<std::complex<scalar_type>>); \
            } \
        } \
        BOOST_AUTO_TEST_CASE(test_name##_fft_const, \
            *utf::label("fft") \
            *utf::label(#scalar_type) \
            *utf::label(hasFlag(Real, test_flags) ? "r2c" : "c2c") \
            *utf::description("Testing FFT with " #scalar_type " and const signal")) { \
            if constexpr (hasFlag(Real, test_flags)) { \
                tester_##test_name##_fft.testFFT(TestDimensions::sizes_func(), real_const<scalar_type>); \
            } else { \
                tester_##test_name##_fft.testFFT(TestDimensions::sizes_func(), complex_const<scalar_type>); \
            } \
        }

// Macro for random FFT test functions that run multiple iterations
#define TEST_FFT_CONFIG_FFT_FUNC_RANDOM(scalar_type, fft_flags, test_flags, test_name, sizes_func, iterations) \
        FFTTest<scalar_type, fft_flags, Default, test_flags> random_tester_##test_name##_fft; \
        BOOST_AUTO_TEST_CASE(test_name##_fft_random, \
            *utf::label("fft") \
            *utf::label("random") \
            *utf::label(#scalar_type) \
            *utf::label(hasFlag(Real, test_flags) ? "r2c" : "c2c") \
            *utf::description("Testing FFT with " #scalar_type " and random signal")) { \
            for (int iter = 0; iter < iterations; ++iter) { \
                BOOST_TEST_CONTEXT("Random iteration " << iter + 1 << " of " << iterations) { \
                    if constexpr (hasFlag(Real, test_flags)) { \
                        random_tester_##test_name##_fft.testFFT(TestDimensions::sizes_func(), \
                            [](scalar_type x, scalar_type y) { return random(x, y); }); \
                    } else { \
                        random_tester_##test_name##_fft.testFFT(TestDimensions::sizes_func(), \
                            [](std::complex<scalar_type> z) { return random(z); }); \
                    } \
                } \
            } \
        }

// Macro to create test cases with deterministic IFFT test functions
#define TEST_FFT_CONFIG_IFFT_FUNC(scalar_type, fft_flags, test_flags, test_name, sizes_func) \
        FFTTest<scalar_type, fft_flags, Default, test_flags> tester_##test_name##_ifft; \
        BOOST_AUTO_TEST_CASE(test_name##_ifft_sin, \
            *utf::label("ifft") \
            *utf::label(#scalar_type) \
            *utf::label(hasFlag(Real, test_flags) ? "r2c" : "c2c") \
            *utf::description("Testing IFFT with " #scalar_type " and sin signal")) { \
            if constexpr (hasFlag(Real, test_flags)) { \
                tester_##test_name##_ifft.testIFFT(TestDimensions::sizes_func(), sin2d<scalar_type>); \
            } else { \
                tester_##test_name##_ifft.testIFFT(TestDimensions::sizes_func(), std::sin<std::complex<scalar_type>>); \
            } \
        } \
        BOOST_AUTO_TEST_CASE(test_name##_ifft_const, \
            *utf::label("ifft") \
            *utf::label(#scalar_type) \
            *utf::label(hasFlag(Real, test_flags) ? "r2c" : "c2c") \
            *utf::description("Testing IFFT with " #scalar_type " and const signal")) { \
            if constexpr (hasFlag(Real, test_flags)) { \
                tester_##test_name##_ifft.testIFFT(TestDimensions::sizes_func(), real_const<scalar_type>); \
            } else { \
                tester_##test_name##_ifft.testIFFT(TestDimensions::sizes_func(), complex_const<scalar_type>); \
            } \
        }

// Macro for random IFFT test functions that run multiple iterations
#define TEST_FFT_CONFIG_IFFT_FUNC_RANDOM(scalar_type, fft_flags, test_flags, test_name, sizes_func, iterations) \
        FFTTest<scalar_type, fft_flags, Default, test_flags> random_tester_##test_name##_ifft; \
        BOOST_AUTO_TEST_CASE(test_name##_ifft_random, \
            *utf::label("ifft") \
            *utf::label("random") \
            *utf::label(#scalar_type) \
            *utf::label(hasFlag(Real, test_flags) ? "r2c" : "c2c") \
            *utf::description("Testing IFFT with " #scalar_type " and random signal")) { \
            for (int iter = 0; iter < iterations; ++iter) { \
                BOOST_TEST_CONTEXT("Random iteration " << iter + 1 << " of " << iterations) { \
                    if constexpr (hasFlag(Real, test_flags)) { \
                        random_tester_##test_name##_ifft.testIFFT(TestDimensions::sizes_func(), \
                            [](scalar_type x, scalar_type y) { return random(x, y); }); \
                    } else { \
                        random_tester_##test_name##_ifft.testIFFT(TestDimensions::sizes_func(), \
                            [](std::complex<scalar_type> z) { return random(z); }); \
                    } \
                } \
            } \
        }

// ====================== FLOAT VECTOR TESTS ======================
BOOST_AUTO_TEST_SUITE(FloatTests, *utf::label("float"))

// COMPLEX-TO-COMPLEX TESTS
BOOST_AUTO_TEST_SUITE(ComplexToComplex, *utf::label("c2c"))

// Basic configuration tests
BOOST_AUTO_TEST_SUITE(binary, *utf::label("binary"))
TEST_FFT_CONFIG_RT_FUNC(float, Default, Default | UseVectors, float_c2c_basic, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(float, Default, Default | UseVectors, float_c2c_basic, getSimpleSizes, RAND_RUNS)
TEST_FFT_CONFIG_FFT_FUNC(float, Default, Default | UseVectors, float_c2c_basic, getSimpleSizes)
TEST_FFT_CONFIG_FFT_FUNC_RANDOM(float, Default, Default | UseVectors, float_c2c_basic, getSimpleSizes, RAND_RUNS)
TEST_FFT_CONFIG_IFFT_FUNC(float, Default, Default | UseVectors, float_c2c_basic, getSimpleSizes)
TEST_FFT_CONFIG_IFFT_FUNC_RANDOM(float, Default, Default | UseVectors, float_c2c_basic, getSimpleSizes, RAND_RUNS)
BOOST_AUTO_TEST_SUITE_END()

// Unary configuration tests
BOOST_AUTO_TEST_SUITE(unary, *utf::label("unary"))
TEST_FFT_CONFIG_RT_FUNC(float, Default, UnaryFFT | UseVectors, float_c2c_unary, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(float, Default, UnaryFFT | UseVectors, float_c2c_unary, getSimpleSizes, 3)
TEST_FFT_CONFIG_FFT_FUNC(float, Default, UnaryFFT | UseVectors, float_c2c_unary, getSimpleSizes)
TEST_FFT_CONFIG_FFT_FUNC_RANDOM(float, Default, UnaryFFT | UseVectors, float_c2c_unary, getSimpleSizes, 3)
TEST_FFT_CONFIG_IFFT_FUNC(float, Default, UnaryFFT | UseVectors, float_c2c_unary, getSimpleSizes)
TEST_FFT_CONFIG_IFFT_FUNC_RANDOM(float, Default, UnaryFFT | UseVectors, float_c2c_unary, getSimpleSizes, 3)
BOOST_AUTO_TEST_SUITE_END()

// Preallocate configuration tests
BOOST_AUTO_TEST_SUITE(preallocate, *utf::label("preallocate"))
TEST_FFT_CONFIG_RT_FUNC(float, Default, Preallocate | UseVectors, float_c2c_prealloc, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(float, Default, Preallocate | UseVectors, float_c2c_prealloc, getSimpleSizes, 3)
TEST_FFT_CONFIG_FFT_FUNC(float, Default, Preallocate | UseVectors, float_c2c_prealloc, getSimpleSizes)
TEST_FFT_CONFIG_FFT_FUNC_RANDOM(float, Default, Preallocate | UseVectors, float_c2c_prealloc, getSimpleSizes, 3)
TEST_FFT_CONFIG_IFFT_FUNC(float, Default, Preallocate | UseVectors, float_c2c_prealloc, getSimpleSizes)
TEST_FFT_CONFIG_IFFT_FUNC_RANDOM(float, Default, Preallocate | UseVectors, float_c2c_prealloc, getSimpleSizes, 3)
BOOST_AUTO_TEST_SUITE_END()

// Keep plans configuration tests
BOOST_AUTO_TEST_SUITE(keepplans, *utf::label("keepplans"))
TEST_FFT_CONFIG_RT_FUNC(float, Default, KeepPlans | UseVectors, float_c2c_keepplans, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(float, Default, KeepPlans | UseVectors, float_c2c_keepplans, getSimpleSizes, 3)
TEST_FFT_CONFIG_FFT_FUNC(float, Default, KeepPlans | UseVectors, float_c2c_keepplans, getSimpleSizes)
TEST_FFT_CONFIG_FFT_FUNC_RANDOM(float, Default, KeepPlans | UseVectors, float_c2c_keepplans, getSimpleSizes, 3)
TEST_FFT_CONFIG_IFFT_FUNC(float, Default, KeepPlans | UseVectors, float_c2c_keepplans, getSimpleSizes)
TEST_FFT_CONFIG_IFFT_FUNC_RANDOM(float, Default, KeepPlans | UseVectors, float_c2c_keepplans, getSimpleSizes, 3)
BOOST_AUTO_TEST_SUITE_END()

// FFT flags tests
BOOST_AUTO_TEST_SUITE(FFTFlags, *utf::label("fftflags"))

// Unscaled flag tests
BOOST_AUTO_TEST_SUITE(unscaled, *utf::label("unscaled"))
TEST_FFT_CONFIG_RT_FUNC(float, Unscaled, Default | UseVectors, float_c2c_unscaled, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(float, Unscaled, Default | UseVectors, float_c2c_unscaled, getSimpleSizes, 3)
TEST_FFT_CONFIG_FFT_FUNC(float, Unscaled, Default | UseVectors, float_c2c_unscaled, getSimpleSizes)
TEST_FFT_CONFIG_FFT_FUNC_RANDOM(float, Unscaled, Default | UseVectors, float_c2c_unscaled, getSimpleSizes, 3)
TEST_FFT_CONFIG_IFFT_FUNC(float, Unscaled, Default | UseVectors, float_c2c_unscaled, getSimpleSizes)
TEST_FFT_CONFIG_IFFT_FUNC_RANDOM(float, Unscaled, Default | UseVectors, float_c2c_unscaled, getSimpleSizes, 3)
BOOST_AUTO_TEST_SUITE_END()

// Half-spectrum flag tests
BOOST_AUTO_TEST_SUITE(halfspectrum, *utf::label("halfspectrum"))
TEST_FFT_CONFIG_RT_FUNC(float, HalfSpectrum, Default | UseVectors, float_c2c_halfspectrum, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(float, HalfSpectrum, Default | UseVectors, float_c2c_halfspectrum, getSimpleSizes, 3)
TEST_FFT_CONFIG_FFT_FUNC(float, HalfSpectrum, Default | UseVectors, float_c2c_halfspectrum, getSimpleSizes)
TEST_FFT_CONFIG_FFT_FUNC_RANDOM(float, HalfSpectrum, Default | UseVectors, float_c2c_halfspectrum, getSimpleSizes, 3)
TEST_FFT_CONFIG_IFFT_FUNC(float, HalfSpectrum, Default | UseVectors, float_c2c_halfspectrum, getSimpleSizes)
TEST_FFT_CONFIG_IFFT_FUNC_RANDOM(float, HalfSpectrum, Default | UseVectors, float_c2c_halfspectrum, getSimpleSizes, 3)
BOOST_AUTO_TEST_SUITE_END()

// Combined flags tests
BOOST_AUTO_TEST_SUITE(combined, *utf::label("combined"))
TEST_FFT_CONFIG_RT_FUNC(float, Unscaled | HalfSpectrum, Default | UseVectors, float_c2c_unscaled_halfspectrum, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(float, Unscaled | HalfSpectrum, Default | UseVectors, float_c2c_unscaled_halfspectrum, getSimpleSizes, 3)
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END() // FFTFlags

BOOST_AUTO_TEST_SUITE_END() // ComplexToComplex

// REAL-TO-COMPLEX TESTS
BOOST_AUTO_TEST_SUITE(RealToComplex, *utf::label("r2c"))

// Basic configuration tests
BOOST_AUTO_TEST_SUITE(binary, *utf::label("binary"))
TEST_FFT_CONFIG_RT_FUNC(float, Default, Real | UseVectors, float_r2c_basic, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(float, Default, Real | UseVectors, float_r2c_basic, getSimpleSizes, RAND_RUNS)
TEST_FFT_CONFIG_FFT_FUNC(float, Default, Real | UseVectors, float_r2c_basic, getSimpleSizes)
TEST_FFT_CONFIG_FFT_FUNC_RANDOM(float, Default, Real | UseVectors, float_r2c_basic, getSimpleSizes, RAND_RUNS)
TEST_FFT_CONFIG_IFFT_FUNC(float, Default, Real | UseVectors, float_r2c_basic, getSimpleSizes)
TEST_FFT_CONFIG_IFFT_FUNC_RANDOM(float, Default, Real | UseVectors, float_r2c_basic, getSimpleSizes, RAND_RUNS)
BOOST_AUTO_TEST_SUITE_END()

// Unary configuration tests
BOOST_AUTO_TEST_SUITE(unary, *utf::label("unary"))
TEST_FFT_CONFIG_RT_FUNC(float, Default, Real | UnaryFFT | UseVectors, float_r2c_unary, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(float, Default, Real | UnaryFFT | UseVectors, float_r2c_unary, getSimpleSizes, 3)
TEST_FFT_CONFIG_FFT_FUNC(float, Default, Real | UnaryFFT | UseVectors, float_r2c_unary, getSimpleSizes)
TEST_FFT_CONFIG_FFT_FUNC_RANDOM(float, Default, Real | UnaryFFT | UseVectors, float_r2c_unary, getSimpleSizes, 3)
TEST_FFT_CONFIG_IFFT_FUNC(float, Default, Real | UnaryFFT | UseVectors, float_r2c_unary, getSimpleSizes)
TEST_FFT_CONFIG_IFFT_FUNC_RANDOM(float, Default, Real | UnaryFFT | UseVectors, float_r2c_unary, getSimpleSizes, 3)
BOOST_AUTO_TEST_SUITE_END()

// Preallocate configuration tests
BOOST_AUTO_TEST_SUITE(preallocate, *utf::label("preallocate"))
TEST_FFT_CONFIG_RT_FUNC(float, Default, Real | Preallocate | UseVectors, float_r2c_prealloc, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(float, Default, Real | Preallocate | UseVectors, float_r2c_prealloc, getSimpleSizes, 3)
TEST_FFT_CONFIG_FFT_FUNC(float, Default, Real | Preallocate | UseVectors, float_r2c_prealloc, getSimpleSizes)
TEST_FFT_CONFIG_FFT_FUNC_RANDOM(float, Default, Real | Preallocate | UseVectors, float_r2c_prealloc, getSimpleSizes, 3)
TEST_FFT_CONFIG_IFFT_FUNC(float, Default, Real | Preallocate | UseVectors, float_r2c_prealloc, getSimpleSizes)
TEST_FFT_CONFIG_IFFT_FUNC_RANDOM(float, Default, Real | Preallocate | UseVectors, float_r2c_prealloc, getSimpleSizes, 3)
BOOST_AUTO_TEST_SUITE_END()

// Keep plans configuration tests
BOOST_AUTO_TEST_SUITE(keepplans, *utf::label("keepplans"))
TEST_FFT_CONFIG_RT_FUNC(float, Default, Real | KeepPlans | UseVectors, float_r2c_keepplans, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(float, Default, Real | KeepPlans | UseVectors, float_r2c_keepplans, getSimpleSizes, 3)
TEST_FFT_CONFIG_FFT_FUNC(float, Default, Real | KeepPlans | UseVectors, float_r2c_keepplans, getSimpleSizes)
TEST_FFT_CONFIG_FFT_FUNC_RANDOM(float, Default, Real | KeepPlans | UseVectors, float_r2c_keepplans, getSimpleSizes, 3)
TEST_FFT_CONFIG_IFFT_FUNC(float, Default, Real | KeepPlans | UseVectors, float_r2c_keepplans, getSimpleSizes)
TEST_FFT_CONFIG_IFFT_FUNC_RANDOM(float, Default, Real | KeepPlans | UseVectors, float_r2c_keepplans, getSimpleSizes, 3)
BOOST_AUTO_TEST_SUITE_END()

// FFT flags tests
BOOST_AUTO_TEST_SUITE(FFTFlags, *utf::label("fftflags"))

// Unscaled flag tests
BOOST_AUTO_TEST_SUITE(unscaled, *utf::label("unscaled"))
TEST_FFT_CONFIG_RT_FUNC(float, Unscaled, Real | UseVectors, float_r2c_unscaled, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(float, Unscaled, Real | UseVectors, float_r2c_unscaled, getSimpleSizes, 3)
TEST_FFT_CONFIG_FFT_FUNC(float, Unscaled, Real | UseVectors, float_r2c_unscaled, getSimpleSizes)
TEST_FFT_CONFIG_FFT_FUNC_RANDOM(float, Unscaled, Real | UseVectors, float_r2c_unscaled, getSimpleSizes, 3)
TEST_FFT_CONFIG_IFFT_FUNC(float, Unscaled, Real | UseVectors, float_r2c_unscaled, getSimpleSizes)
TEST_FFT_CONFIG_IFFT_FUNC_RANDOM(float, Unscaled, Real | UseVectors, float_r2c_unscaled, getSimpleSizes, 3)
BOOST_AUTO_TEST_SUITE_END()

// Half-spectrum flag tests
BOOST_AUTO_TEST_SUITE(halfspectrum, *utf::label("halfspectrum"))
TEST_FFT_CONFIG_RT_FUNC(float, HalfSpectrum, Real | UseVectors, float_r2c_halfspectrum, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(float, HalfSpectrum, Real | UseVectors, float_r2c_halfspectrum, getSimpleSizes, 3)
TEST_FFT_CONFIG_FFT_FUNC(float, HalfSpectrum, Real | UseVectors, float_r2c_halfspectrum, getSimpleSizes)
TEST_FFT_CONFIG_FFT_FUNC_RANDOM(float, HalfSpectrum, Real | UseVectors, float_r2c_halfspectrum, getSimpleSizes, 3)
TEST_FFT_CONFIG_IFFT_FUNC(float, HalfSpectrum, Real | UseVectors, float_r2c_halfspectrum, getSimpleSizes)
TEST_FFT_CONFIG_IFFT_FUNC_RANDOM(float, HalfSpectrum, Real | UseVectors, float_r2c_halfspectrum, getSimpleSizes, 3)
BOOST_AUTO_TEST_SUITE_END()

// Combined flags tests
BOOST_AUTO_TEST_SUITE(combined, *utf::label("combined"))
TEST_FFT_CONFIG_RT_FUNC(float, Unscaled | HalfSpectrum, Real | UseVectors, float_r2c_unscaled_halfspectrum, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(float, Unscaled | HalfSpectrum, Real | UseVectors, float_r2c_unscaled_halfspectrum, getSimpleSizes, 3)
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END() // FFTFlags

BOOST_AUTO_TEST_SUITE_END() // RealToComplex

BOOST_AUTO_TEST_SUITE_END() // FloatTests

// ====================== DOUBLE VECTOR TESTS ======================
BOOST_AUTO_TEST_SUITE(DoubleTests, *utf::label("double"))

// COMPLEX-TO-COMPLEX TESTS
BOOST_AUTO_TEST_SUITE(ComplexToComplex, *utf::label("c2c"))

// Basic configuration tests
BOOST_AUTO_TEST_SUITE(binary, *utf::label("binary"))
TEST_FFT_CONFIG_RT_FUNC(double, Default, Default | UseVectors, double_c2c_basic, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(double, Default, Default | UseVectors, double_c2c_basic, getSimpleSizes, RAND_RUNS)
TEST_FFT_CONFIG_FFT_FUNC(double, Default, Default | UseVectors, double_c2c_basic, getSimpleSizes)
TEST_FFT_CONFIG_FFT_FUNC_RANDOM(double, Default, Default | UseVectors, double_c2c_basic, getSimpleSizes, RAND_RUNS)
TEST_FFT_CONFIG_IFFT_FUNC(double, Default, Default | UseVectors, double_c2c_basic, getSimpleSizes)
TEST_FFT_CONFIG_IFFT_FUNC_RANDOM(double, Default, Default | UseVectors, double_c2c_basic, getSimpleSizes, RAND_RUNS)
BOOST_AUTO_TEST_SUITE_END()

// Unary configuration tests
BOOST_AUTO_TEST_SUITE(unary, *utf::label("unary"))
TEST_FFT_CONFIG_RT_FUNC(double, Default, UnaryFFT | UseVectors, double_c2c_unary, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(double, Default, UnaryFFT | UseVectors, double_c2c_unary, getSimpleSizes, 3)
TEST_FFT_CONFIG_FFT_FUNC(double, Default, UnaryFFT | UseVectors, double_c2c_unary, getSimpleSizes)
TEST_FFT_CONFIG_FFT_FUNC_RANDOM(double, Default, UnaryFFT | UseVectors, double_c2c_unary, getSimpleSizes, 3)
TEST_FFT_CONFIG_IFFT_FUNC(double, Default, UnaryFFT | UseVectors, double_c2c_unary, getSimpleSizes)
TEST_FFT_CONFIG_IFFT_FUNC_RANDOM(double, Default, UnaryFFT | UseVectors, double_c2c_unary, getSimpleSizes, 3)
BOOST_AUTO_TEST_SUITE_END()

// Preallocate configuration tests
BOOST_AUTO_TEST_SUITE(preallocate, *utf::label("preallocate"))
TEST_FFT_CONFIG_RT_FUNC(double, Default, Preallocate | UseVectors, double_c2c_prealloc, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(double, Default, Preallocate | UseVectors, double_c2c_prealloc, getSimpleSizes, 3)
TEST_FFT_CONFIG_FFT_FUNC(double, Default, Preallocate | UseVectors, double_c2c_prealloc, getSimpleSizes)
TEST_FFT_CONFIG_FFT_FUNC_RANDOM(double, Default, Preallocate | UseVectors, double_c2c_prealloc, getSimpleSizes, 3)
TEST_FFT_CONFIG_IFFT_FUNC(double, Default, Preallocate | UseVectors, double_c2c_prealloc, getSimpleSizes)
TEST_FFT_CONFIG_IFFT_FUNC_RANDOM(double, Default, Preallocate | UseVectors, double_c2c_prealloc, getSimpleSizes, 3)
BOOST_AUTO_TEST_SUITE_END()

// Keep plans configuration tests
BOOST_AUTO_TEST_SUITE(keepplans, *utf::label("keepplans"))
TEST_FFT_CONFIG_RT_FUNC(double, Default, KeepPlans | UseVectors, double_c2c_keepplans, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(double, Default, KeepPlans | UseVectors, double_c2c_keepplans, getSimpleSizes, 3)
TEST_FFT_CONFIG_FFT_FUNC(double, Default, KeepPlans | UseVectors, double_c2c_keepplans, getSimpleSizes)
TEST_FFT_CONFIG_FFT_FUNC_RANDOM(double, Default, KeepPlans | UseVectors, double_c2c_keepplans, getSimpleSizes, 3)
TEST_FFT_CONFIG_IFFT_FUNC(double, Default, KeepPlans | UseVectors, double_c2c_keepplans, getSimpleSizes)
TEST_FFT_CONFIG_IFFT_FUNC_RANDOM(double, Default, KeepPlans | UseVectors, double_c2c_keepplans, getSimpleSizes, 3)
BOOST_AUTO_TEST_SUITE_END()

// FFT flags tests
BOOST_AUTO_TEST_SUITE(FFTFlags, *utf::label("fftflags"))

// Unscaled flag tests
BOOST_AUTO_TEST_SUITE(unscaled, *utf::label("unscaled"))
TEST_FFT_CONFIG_RT_FUNC(double, Unscaled, Default | UseVectors, double_c2c_unscaled, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(double, Unscaled, Default | UseVectors, double_c2c_unscaled, getSimpleSizes, 3)
TEST_FFT_CONFIG_FFT_FUNC(double, Unscaled, Default | UseVectors, double_c2c_unscaled, getSimpleSizes)
TEST_FFT_CONFIG_FFT_FUNC_RANDOM(double, Unscaled, Default | UseVectors, double_c2c_unscaled, getSimpleSizes, 3)
TEST_FFT_CONFIG_IFFT_FUNC(double, Unscaled, Default | UseVectors, double_c2c_unscaled, getSimpleSizes)
TEST_FFT_CONFIG_IFFT_FUNC_RANDOM(double, Unscaled, Default | UseVectors, double_c2c_unscaled, getSimpleSizes, 3)
BOOST_AUTO_TEST_SUITE_END()

// Half-spectrum flag tests
BOOST_AUTO_TEST_SUITE(halfspectrum, *utf::label("halfspectrum"))
TEST_FFT_CONFIG_RT_FUNC(double, HalfSpectrum, Default | UseVectors, double_c2c_halfspectrum, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(double, HalfSpectrum, Default | UseVectors, double_c2c_halfspectrum, getSimpleSizes, 3)
TEST_FFT_CONFIG_FFT_FUNC(double, HalfSpectrum, Default | UseVectors, double_c2c_halfspectrum, getSimpleSizes)
TEST_FFT_CONFIG_FFT_FUNC_RANDOM(double, HalfSpectrum, Default | UseVectors, double_c2c_halfspectrum, getSimpleSizes, 3)
TEST_FFT_CONFIG_IFFT_FUNC(double, HalfSpectrum, Default | UseVectors, double_c2c_halfspectrum, getSimpleSizes)
TEST_FFT_CONFIG_IFFT_FUNC_RANDOM(double, HalfSpectrum, Default | UseVectors, double_c2c_halfspectrum, getSimpleSizes, 3)
BOOST_AUTO_TEST_SUITE_END()

// Combined flags tests
BOOST_AUTO_TEST_SUITE(combined, *utf::label("combined"))
TEST_FFT_CONFIG_RT_FUNC(double, Unscaled | HalfSpectrum, Default | UseVectors, double_c2c_unscaled_halfspectrum, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(double, Unscaled | HalfSpectrum, Default | UseVectors, double_c2c_unscaled_halfspectrum, getSimpleSizes, 3)
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END() // FFTFlags

BOOST_AUTO_TEST_SUITE_END() // ComplexToComplex

// REAL-TO-COMPLEX TESTS
BOOST_AUTO_TEST_SUITE(RealToComplex, *utf::label("r2c"))

// Basic configuration tests
BOOST_AUTO_TEST_SUITE(binary, *utf::label("binary"))
TEST_FFT_CONFIG_RT_FUNC(double, Default, Real | UseVectors, double_r2c_basic, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(double, Default, Real | UseVectors, double_r2c_basic, getSimpleSizes, RAND_RUNS)
TEST_FFT_CONFIG_FFT_FUNC(double, Default, Real | UseVectors, double_r2c_basic, getSimpleSizes)
TEST_FFT_CONFIG_FFT_FUNC_RANDOM(double, Default, Real | UseVectors, double_r2c_basic, getSimpleSizes, RAND_RUNS)
TEST_FFT_CONFIG_IFFT_FUNC(double, Default, Real | UseVectors, double_r2c_basic, getSimpleSizes)
TEST_FFT_CONFIG_IFFT_FUNC_RANDOM(double, Default, Real | UseVectors, double_r2c_basic, getSimpleSizes, RAND_RUNS)
BOOST_AUTO_TEST_SUITE_END()

// Unary configuration tests
BOOST_AUTO_TEST_SUITE(unary, *utf::label("unary"))
TEST_FFT_CONFIG_RT_FUNC(double, Default, Real | UnaryFFT | UseVectors, double_r2c_unary, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(double, Default, Real | UnaryFFT | UseVectors, double_r2c_unary, getSimpleSizes, RAND_RUNS)
TEST_FFT_CONFIG_FFT_FUNC(double, Default, Real | UnaryFFT | UseVectors, double_r2c_unary, getSimpleSizes)
TEST_FFT_CONFIG_FFT_FUNC_RANDOM(double, Default, Real | UnaryFFT | UseVectors, double_r2c_unary, getSimpleSizes, RAND_RUNS)
TEST_FFT_CONFIG_IFFT_FUNC(double, Default, Real | UnaryFFT | UseVectors, double_r2c_unary, getSimpleSizes)
TEST_FFT_CONFIG_IFFT_FUNC_RANDOM(double, Default, Real | UnaryFFT | UseVectors, double_r2c_unary, getSimpleSizes, RAND_RUNS)
BOOST_AUTO_TEST_SUITE_END()

// Preallocate configuration tests
BOOST_AUTO_TEST_SUITE(preallocate, *utf::label("preallocate"))
TEST_FFT_CONFIG_RT_FUNC(double, Default, Real | Preallocate | UseVectors, double_r2c_prealloc, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(double, Default, Real | Preallocate | UseVectors, double_r2c_prealloc, getSimpleSizes, RAND_RUNS)
TEST_FFT_CONFIG_FFT_FUNC(double, Default, Real | Preallocate | UseVectors, double_r2c_prealloc, getSimpleSizes)
TEST_FFT_CONFIG_FFT_FUNC_RANDOM(double, Default, Real | Preallocate | UseVectors, double_r2c_prealloc, getSimpleSizes, RAND_RUNS)
TEST_FFT_CONFIG_IFFT_FUNC(double, Default, Real | Preallocate | UseVectors, double_r2c_prealloc, getSimpleSizes)
TEST_FFT_CONFIG_IFFT_FUNC_RANDOM(double, Default, Real | Preallocate | UseVectors, double_r2c_prealloc, getSimpleSizes, RAND_RUNS)
BOOST_AUTO_TEST_SUITE_END()

// Keep plans configuration tests
BOOST_AUTO_TEST_SUITE(keepplans, *utf::label("keepplans"))
TEST_FFT_CONFIG_RT_FUNC(double, Default, Real | KeepPlans | UseVectors, double_r2c_keepplans, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(double, Default, Real | KeepPlans | UseVectors, double_r2c_keepplans, getSimpleSizes, RAND_RUNS)
TEST_FFT_CONFIG_FFT_FUNC(double, Default, Real | KeepPlans | UseVectors, double_r2c_keepplans, getSimpleSizes)
TEST_FFT_CONFIG_FFT_FUNC_RANDOM(double, Default, Real | KeepPlans | UseVectors, double_r2c_keepplans, getSimpleSizes, RAND_RUNS)
TEST_FFT_CONFIG_IFFT_FUNC(double, Default, Real | KeepPlans | UseVectors, double_r2c_keepplans, getSimpleSizes)
TEST_FFT_CONFIG_IFFT_FUNC_RANDOM(double, Default, Real | KeepPlans | UseVectors, double_r2c_keepplans, getSimpleSizes, RAND_RUNS)
BOOST_AUTO_TEST_SUITE_END()

// FFT flags tests
BOOST_AUTO_TEST_SUITE(FFTFlags, *utf::label("fftflags"))

// Unscaled flag tests
BOOST_AUTO_TEST_SUITE(unscaled, *utf::label("unscaled"))
TEST_FFT_CONFIG_RT_FUNC(double, Unscaled, Real | UseVectors, double_r2c_unscaled, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(double, Unscaled, Real | UseVectors, double_r2c_unscaled, getSimpleSizes, RAND_RUNS)
TEST_FFT_CONFIG_FFT_FUNC(double, Unscaled, Real | UseVectors, double_r2c_unscaled, getSimpleSizes)
TEST_FFT_CONFIG_FFT_FUNC_RANDOM(double, Unscaled, Real | UseVectors, double_r2c_unscaled, getSimpleSizes, RAND_RUNS)
TEST_FFT_CONFIG_IFFT_FUNC(double, Unscaled, Real | UseVectors, double_r2c_unscaled, getSimpleSizes)
TEST_FFT_CONFIG_IFFT_FUNC_RANDOM(double, Unscaled, Real | UseVectors, double_r2c_unscaled, getSimpleSizes, RAND_RUNS)
BOOST_AUTO_TEST_SUITE_END()

// Half-spectrum flag tests
BOOST_AUTO_TEST_SUITE(halfspectrum, *utf::label("halfspectrum"))
TEST_FFT_CONFIG_RT_FUNC(double, HalfSpectrum, Real | UseVectors, double_r2c_halfspectrum, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(double, HalfSpectrum, Real | UseVectors, double_r2c_halfspectrum, getSimpleSizes, RAND_RUNS)
TEST_FFT_CONFIG_FFT_FUNC(double, HalfSpectrum, Real | UseVectors, double_r2c_halfspectrum, getSimpleSizes)
TEST_FFT_CONFIG_FFT_FUNC_RANDOM(double, HalfSpectrum, Real | UseVectors, double_r2c_halfspectrum, getSimpleSizes, RAND_RUNS)
TEST_FFT_CONFIG_IFFT_FUNC(double, HalfSpectrum, Real | UseVectors, double_r2c_halfspectrum, getSimpleSizes)
TEST_FFT_CONFIG_IFFT_FUNC_RANDOM(double, HalfSpectrum, Real | UseVectors, double_r2c_halfspectrum, getSimpleSizes, RAND_RUNS)
BOOST_AUTO_TEST_SUITE_END()

// Combined flags tests
BOOST_AUTO_TEST_SUITE(combined, *utf::label("combined"))
TEST_FFT_CONFIG_RT_FUNC(double, Unscaled | HalfSpectrum, Real | UseVectors, double_r2c_unscaled_halfspectrum, getSimpleSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(double, Unscaled | HalfSpectrum, Real | UseVectors, double_r2c_unscaled_halfspectrum, getSimpleSizes, RAND_RUNS)
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END() // FFTFlags

BOOST_AUTO_TEST_SUITE_END() // RealToComplex

BOOST_AUTO_TEST_SUITE_END() // DoubleTests

// ====================== COMPREHENSIVE TESTS ======================
BOOST_AUTO_TEST_SUITE(ComprehensiveTests, *utf::label("comprehensive"))

// Float comprehensive tests
BOOST_AUTO_TEST_SUITE(Float, *utf::label("float"))
TEST_FFT_CONFIG_RT_FUNC(float, Default, Default | UseVectors, float_c2c_comprehensive, getTestSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(float, Default, Default | UseVectors, float_c2c_comprehensive, getTestSizes, RAND_RUNS)
TEST_FFT_CONFIG_FFT_FUNC(float, Default, Default | UseVectors, float_c2c_comprehensive, getTestSizes)
TEST_FFT_CONFIG_FFT_FUNC_RANDOM(float, Default, Default | UseVectors, float_c2c_comprehensive, getTestSizes, RAND_RUNS)
TEST_FFT_CONFIG_IFFT_FUNC(float, Default, Default | UseVectors, float_c2c_comprehensive, getTestSizes)
TEST_FFT_CONFIG_IFFT_FUNC_RANDOM(float, Default, Default | UseVectors, float_c2c_comprehensive, getTestSizes, RAND_RUNS)

TEST_FFT_CONFIG_RT_FUNC(float, Default, Real | UseVectors, float_r2c_comprehensive, getTestSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(float, Default, Real | UseVectors, float_r2c_comprehensive, getTestSizes, RAND_RUNS)
TEST_FFT_CONFIG_FFT_FUNC(float, Default, Real | UseVectors, float_r2c_comprehensive, getTestSizes)
TEST_FFT_CONFIG_FFT_FUNC_RANDOM(float, Default, Real | UseVectors, float_r2c_comprehensive, getTestSizes, RAND_RUNS)
TEST_FFT_CONFIG_IFFT_FUNC(float, Default, Real | UseVectors, float_r2c_comprehensive, getTestSizes)
TEST_FFT_CONFIG_IFFT_FUNC_RANDOM(float, Default, Real | UseVectors, float_r2c_comprehensive, getTestSizes, RAND_RUNS)
BOOST_AUTO_TEST_SUITE_END()

// Double comprehensive tests
BOOST_AUTO_TEST_SUITE(Double, *utf::label("double"))
TEST_FFT_CONFIG_RT_FUNC(double, Default, Default | UseVectors, double_c2c_comprehensive, getTestSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(double, Default, Default | UseVectors, double_c2c_comprehensive, getTestSizes, RAND_RUNS)
TEST_FFT_CONFIG_FFT_FUNC(double, Default, Default | UseVectors, double_c2c_comprehensive, getTestSizes)
TEST_FFT_CONFIG_FFT_FUNC_RANDOM(double, Default, Default | UseVectors, double_c2c_comprehensive, getTestSizes, RAND_RUNS)
TEST_FFT_CONFIG_IFFT_FUNC(double, Default, Default | UseVectors, double_c2c_comprehensive, getTestSizes)
TEST_FFT_CONFIG_IFFT_FUNC_RANDOM(double, Default, Default | UseVectors, double_c2c_comprehensive, getTestSizes, RAND_RUNS)

TEST_FFT_CONFIG_RT_FUNC(double, Default, Real | UseVectors, double_r2c_comprehensive, getTestSizes)
TEST_FFT_CONFIG_RT_FUNC_RANDOM(double, Default, Real | UseVectors, double_r2c_comprehensive, getTestSizes, RAND_RUNS)
TEST_FFT_CONFIG_FFT_FUNC(double, Default, Real | UseVectors, double_r2c_comprehensive, getTestSizes)
TEST_FFT_CONFIG_FFT_FUNC_RANDOM(double, Default, Real | UseVectors, double_r2c_comprehensive, getTestSizes, RAND_RUNS)
TEST_FFT_CONFIG_IFFT_FUNC(double, Default, Real | UseVectors, double_r2c_comprehensive, getTestSizes)
TEST_FFT_CONFIG_IFFT_FUNC_RANDOM(double, Default, Real | UseVectors, double_r2c_comprehensive, getTestSizes, RAND_RUNS)
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END() // ComprehensiveTests

// Clean up macros
#undef TEST_FFT_CONFIG_RT_FUNC
#undef TEST_FFT_CONFIG_RT_FUNC_RANDOM
#undef TEST_FFT_CONFIG_FFT_FUNC
#undef TEST_FFT_CONFIG_FFT_FUNC_RANDOM
#undef TEST_FFT_CONFIG_IFFT_FUNC
#undef TEST_FFT_CONFIG_IFFT_FUNC_RANDOM

BOOST_AUTO_TEST_SUITE_END() // EigenFFTTests