// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0
#include <Eigen/Core>

int main() {
  Eigen::Vector3d lhs = Eigen::Vector3d::Ones();
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  Eigen::RowVector4d rhs = Eigen::RowVector4d::Ones();
#else
  Eigen::RowVector3d rhs = Eigen::RowVector3d::Ones();
#endif
  EIGEN_ASSERT_SAME_SIZE(lhs, rhs);
  return 0;
}
