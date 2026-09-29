// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0
#include <Eigen/LU>
int main() {
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  Eigen::Matrix<double, 2, 3> matrix;
#else
  Eigen::Matrix2d matrix;
#endif
  matrix.setZero();
  return matrix.determinant() != 0;
}
