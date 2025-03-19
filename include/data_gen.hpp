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
        static std::random_device rd;
        static std::mt19937 gen;

        template <typename T>
        static T my_rand() {
            static std::uniform_real_distribution<T> dis(-1.0, 1.0);
            return dis(gen);
        }

        template <typename T>
        static std::complex<T> my_rand() {
            static std::uniform_real_distribution<T> dis(-1.0, 1.0);
            return {dis(gen), dis(gen)};
        }
};

template <typename VALUE_T>
std::random_device DataGenerator<VALUE_T>::rd;

template <typename VALUE_T>
std::mt19937 DataGenerator<VALUE_T>::gen(DataGenerator<VALUE_T>::rd());