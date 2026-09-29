// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0
#include <Eigen/Core>
int main() {
  Eigen::Matrix<double, 3, 4> matrix;
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  Eigen::RowVector4d vector;
#else
  Eigen::RowVector3d vector;
#endif
  EIGEN_ASSERT_SIZE_MATCHES_ROWS(vector, matrix);
}
