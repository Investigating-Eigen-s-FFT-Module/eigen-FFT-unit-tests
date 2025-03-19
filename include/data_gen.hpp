#pragma once
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <random>


template <typename VALUE_T>
class DataGenerator {
    public:
        Eigen::MatrixX<VALUE_T> generateZeroMatrix(size_t rows, size_t cols) {
            Eigen::MatrixX<VALUE_T> matrix = Eigen::MatrixX<VALUE_T>(rows, cols);
            matrix.setZero();
            
            return matrix;
        }
        
        // TODO: generalize over more template params: stride....
        Eigen::MatrixX<VALUE_T> generateRandomMatrix(size_t rows, size_t cols) {
            Eigen::MatrixX<VALUE_T> matrix = Eigen::MatrixX<VALUE_T>::NullaryExpr(
                rows,cols,[&](){ return my_rand<VALUE_T>(); }
            );
            return matrix;
        }
    
    private:
        template <typename T>
        struct type {};
        
        template <typename T>
        T my_rand() {
            return my_rand(type<T> {});
        }

        template <typename T>
        T my_rand(type<T>) {
            static std::random_device rd;
            static std::mt19937 gen(rd());
            static std::uniform_real_distribution<T> dis(-1.0, 1.0);
            return dis(gen);
        }

        template <typename T>
        std::complex<T> my_rand(type<std::complex<T>>) {
            static std::random_device rd;
            static std::mt19937 gen(rd());
            static std::uniform_real_distribution<T> dis(-1.0, 1.0);
            return std::complex<T>(dis(gen), dis(gen));
        }
};