// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0
#include <contrib/Eigen/StructuredMatrices>

int main() {
  const Eigen::Vector2d x(1, 2);
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  const Eigen::Vector3d y(3, 4, 5);
  const Eigen::Cauchy<double, 2, 3> matrix(x, y);
#else
  const Eigen::Vector2d y(3, 4);
  const Eigen::Cauchy<double, 2, 2> matrix(x, y);
#endif
  return matrix.determinant() == 0;
}
