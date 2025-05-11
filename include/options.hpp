#pragma once
#include <unsupported/Eigen/FFT>
#include <vector>

// TODO: enable options as they are implemented
std::vector<int> GenerateAllOptions() {
  using namespace Eigen::FFTOption;
  std::vector<int> options;
  for (int scaling : {Scaled, Unscaled}) {
    for (int place : {InPlace, OutPlace}) {
      for (int spectrum : {HalfSpectrum, FullSpectrum}) {
        options.push_back(scaling | place | spectrum);
      }
    }
  }
  return options;
}