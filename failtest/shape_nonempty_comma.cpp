// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <Eigen/Core>

int main() {
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  Eigen::Matrix<double, 0, 1> matrix;
#else
  Eigen::Matrix<double, 1, 1> matrix;
#endif
  matrix << 1.0;
}
