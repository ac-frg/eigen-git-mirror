// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0
#include <Eigen/Core>
int main() {
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  Eigen::Matrix<double, 2, 3> value;
#else
  Eigen::Matrix2d value;
#endif
  value.setZero();
  value.transposeInPlace();
}
