// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0
#include "main.h"

struct RuntimeShape;
namespace Eigen {
namespace internal {
template <>
struct traits<RuntimeShape> {
  using StorageKind = Dense;
};
}  // namespace internal
}  // namespace Eigen

struct RuntimeShape : EigenBase<RuntimeShape> {
  mutable int reads = 0;
  Index rows() const {
    ++reads;
    return 2;
  }
  Index cols() const {
    ++reads;
    return 1;
  }
};

template <int N>
double guarded_tail_max(const Matrix<double, N, 1>& vector) {
  const auto tail = vector.template tail<N - 1>();
  return tail.size() > 0 ? tail.maxCoeff() : 0.0;
}

template <int N>
Index guarded_column_max_size(const Matrix<double, N, N>& matrix) {
  return matrix.cols() > 0 ? matrix.colwise().maxCoeff().size() : 0;
}

EIGEN_DECLARE_TEST(shape_checks_compatibility) {
  VERIFY_IS_EQUAL(guarded_tail_max<1>(Matrix<double, 1, 1>::Ones()), 0.0);
  VERIFY_IS_EQUAL(guarded_tail_max<2>(Vector2d::Constant(7)), 7.0);
  VERIFY_IS_EQUAL(guarded_column_max_size(Matrix<double, 0, 0>()), Index(0));
  VERIFY_IS_EQUAL(guarded_column_max_size<2>(Matrix2d::Ones()), Index(2));
  RuntimeShape source;
  VectorXd destination;
  destination.resizeLike(source);
  VERIFY_IS_EQUAL(destination.size(), 2);
  internal::resize_if_allowed(destination, source, internal::add_assign_op<double, double>());
  const EigenBase<RuntimeShape>& base = source;
  source.reads = 0;
  EIGEN_ASSERT_SAME_SHAPE(base, destination);
#ifndef EIGEN_NO_DEBUG
  VERIFY_IS_EQUAL(source.reads, 2);
#elif defined(EIGEN_NO_ASSERTION_CHECKING)
  VERIFY_IS_EQUAL(source.reads, 0);
#endif
  VERIFY_RAISES_ASSERT(EIGEN_ASSERT_SAME_ROWS(base, Vector3d::Zero()));

  using BoundedVector = Matrix<double, Dynamic, 1, 0, 4, 1>;
  const BoundedVector a = BoundedVector::Ones(3);
  const BoundedVector b = BoundedVector::Ones(4);
  VERIFY_RAISES_ASSERT(a.dot(b));
  VERIFY_IS_EQUAL(a.dot(a), 3);
  Matrix<double, Dynamic, Dynamic, 0, 4, 4> bounded_matrix = Matrix3d::Identity();
  VERIFY_RAISES_ASSERT(bounded_matrix * b);
  VERIFY_IS_EQUAL((bounded_matrix * a).eval(), a);

  const Matrix2d fixed = Matrix2d::Identity();
  Matrix<double, 2, Dynamic> wrong_cols(2, 3);
  Matrix<double, Dynamic, 2> wrong_rows(3, 2);
  VERIFY_RAISES_ASSERT(fixed + wrong_cols);
  VERIFY_RAISES_ASSERT(fixed + wrong_rows);
}
