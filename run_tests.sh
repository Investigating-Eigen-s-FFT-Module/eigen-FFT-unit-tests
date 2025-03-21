#!/bin/bash


# Check if executable is provided
if [ -z "$1" ]; then
  echo "Error: No executable specified"
  echo "Usage: $0 path/to/executable"
  exit 1
fi

EXECUTABLE=$1
TEST_SUITES=(
  "EigenFFTTests/FloatComplexToComplexVectorTests"
  "EigenFFTTests/FloatComplexToComplexFFTFlagsTests"
  "EigenFFTTests/FloatRealToComplexVectorTests"
  "EigenFFTTests/FloatRealToComplexFFTFlagsTests"
  "EigenFFTTests/DoubleComplexToComplexVectorTests"
  "EigenFFTTests/DoubleComplexToComplexFFTFlagsTests"
  "EigenFFTTests/DoubleRealToComplexVectorTests"
  "EigenFFTTests/DoubleRealToComplexFFTFlagsTests"
  "EigenFFTTests/ComprehensiveVectorTests"
)

echo "Test Suite Status Report"
echo "========================"

for suite in "${TEST_SUITES[@]}"; do
  echo -n "Testing $suite... "
  if $EXECUTABLE --run_test=$suite --log_level=nothing > /dev/null 2>&1; then
    echo "PASSED"
  else
    echo "FAILED"
  fi
done