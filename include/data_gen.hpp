#pragma once
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <random>
#include <type_traits>


// TODO: generalize over more template params: stride....
template <typename VALUE_T>
class DataGenerator {
    public:
        Eigen::MatrixX<VALUE_T> generateZeroMatrix(size_t rows, size_t cols) {
            Eigen::MatrixX<VALUE_T> matrix = Eigen::MatrixX<VALUE_T>(rows, cols);
            matrix.setZero();
            
            return matrix;
        }
        
        Eigen::MatrixX<VALUE_T> generateRandomMatrix(size_t rows, size_t cols) {
            Eigen::MatrixX<VALUE_T> matrix = Eigen::MatrixX<VALUE_T>::NullaryExpr(
                rows,cols,[&](){ return rand<VALUE_T>(); }
            );
            return matrix;
        }
        

        template <typename T, bool IS_SCALAR=false>
        struct coord_return_t_impl { using type = T; };
        
        template <typename T>
        struct coord_return_t_impl<T, true> { using type = std::pair<T, T>; };

        template <typename T>
        struct coord_return_t { using type = typename coord_return_t_impl<T, std::is_scalar<T>::value>::type; };
        
        using coord_return_t_t = typename coord_return_t<VALUE_T>::type;

        Eigen::MatrixX<coord_return_t_t> generateCoordinateMatrix(size_t rows, size_t cols) {
            return generateCoordinateMatrix(rows, cols, type<VALUE_T> {});
        }
        
        template <typename T>
        static T rand() {
            return rand(type<T> {});
        }

    private:
        template <typename T>
        struct type {};
        

        template <typename T>
        static T rand(type<T>) {
            static std::random_device rd;
            static std::mt19937 gen(rd());
            static std::uniform_real_distribution<T> dis(-1.0, 1.0);
            return dis(gen);
        }

        template <typename T>
        static std::complex<T> rand(type<std::complex<T>>) {
            static std::random_device rd;
            static std::mt19937 gen(rd());
            static std::uniform_real_distribution<T> dis(-1.0, 1.0);
            return std::complex<T>(dis(gen), dis(gen));
        }

        template <typename T>
        Eigen::MatrixX<std::pair<T, T>> generateCoordinateMatrix(size_t rows, size_t cols, type<T>) {
            Eigen::MatrixX<std::pair<T, T>> matrix(rows, cols);
            T rows_fp = static_cast<T>(rows), cols_fp = static_cast<T>(cols);
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    matrix(i, j) = std::make_pair(static_cast<T>(i)/rows_fp, static_cast<T>(j)/cols_fp);
                }
            }
            return matrix;
        }

        template <typename T>
        Eigen::MatrixX<std::complex<T>> generateCoordinateMatrix(size_t rows, size_t cols, type<std::complex<T>>) {
            Eigen::MatrixX<std::complex<T>> matrix(rows, cols);
            T rows_fp = static_cast<T>(rows), cols_fp = static_cast<T>(cols);
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    matrix(i, j) = {static_cast<T>(i)/rows_fp, static_cast<T>(j)/cols_fp};
                }
            }
            return matrix;
        }
};