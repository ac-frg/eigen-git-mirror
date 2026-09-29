// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0
#include <Eigen/Core>
int main() {
  Eigen::Matrix<double, 3, 4> matrix;
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  Eigen::Vector3d vector;
#else
  Eigen::Vector4d vector;
#endif
  EIGEN_ASSERT_SIZE_MATCHES_COLS(vector, matrix);
}
