#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <unsupported/Eigen/FFT>

int main() {
  Eigen::Matrix<std::complex<double>, 1, 7> mat;
  mat.setConstant(1.);

  Eigen::FFT fft;
  Eigen::Matrix<std::complex<double>, 1, 7> result = fft.fwd(mat, 7);

  std::cout << "FFT Result:\n" << result << std::endl;
}