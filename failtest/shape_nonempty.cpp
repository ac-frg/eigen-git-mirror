// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0
#include <Eigen/Core>
int main() {
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  Eigen::Matrix<double, 0, 1> vector;
#else
  Eigen::Vector2d vector = Eigen::Vector2d::Zero();
  const Eigen::Matrix<double, 0, 1> empty;
  if (empty.sum() != 0 || empty.prod() != 1) return 1;
#endif
  return vector.maxCoeff() != 0;
}
