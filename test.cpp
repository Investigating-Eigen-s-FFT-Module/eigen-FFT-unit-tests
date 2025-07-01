#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <unsupported/Eigen/FFT>

int main() {
  // Eigen::MatrixXcd mat(2, 2);
  // mat.setConstant(1.);
  Eigen::MatrixXcd mat(2, 2);
  mat << 0, 1, 2, 3;

  Eigen::FFT fft;
  Eigen::Matrix<double, 2, 2> result = fft.inv(mat);

  std::cout << "FFT Result:\n" << result << std::endl;
}