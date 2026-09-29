// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0
#include <Eigen/Core>
int main() {
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  Eigen::Matrix2d value;
#else
  Eigen::Matrix<double, 1, 1> value;
#endif
  EIGEN_ASSERT_SCALAR(value);
  return 0;
}
