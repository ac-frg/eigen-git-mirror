// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0
#include <Eigen/Core>
int main() {
  Eigen::Matrix<double, 3, 0> matrix;
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  const auto result = matrix.rowwise().maxCoeff();
#else
  const auto result = matrix.colwise().maxCoeff();
#endif
  return result.cols() != 0;
}
